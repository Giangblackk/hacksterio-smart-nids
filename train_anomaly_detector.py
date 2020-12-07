import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import json
from datetime import datetime
import math
import numpy as np

import tensorflow.compat.v1 as tf
from tensorflow import keras


def create_auto_encoder(input_size, beta, activation="relu", name=None):
    hidden_layer_size = math.ceil(beta * input_size)
    seq = keras.models.Sequential(
        [
            keras.layers.Conv2D(
                filters=hidden_layer_size, kernel_size=1, activation=activation
            ),
            keras.layers.Conv2D(
                filters=input_size, kernel_size=1, activation=activation
            ),
        ],
        name=name,
    )
    return seq


if __name__ == "__main__":
    batch_size = 512
    n_epoches = 10
    # load dataset
    dataset_file = "capEC2AMAZ-O4EL3NG-172.31.69.26a.pcap.h5"
    dataset = keras.utils.HDF5Matrix(dataset_file, "dataset")

    numpy_dataset = dataset.data[...]

    dataset_max = np.max(numpy_dataset, axis=0)
    dataset_min = np.min(numpy_dataset, axis=0)

    numpy_dataset = (numpy_dataset - dataset_min) / (1e-10 + dataset_max - dataset_min)

    mapper_file = "mapper.json"

    with open(mapper_file, "r") as f:
        feature_mapper = json.load(f)
    print(feature_mapper)

    ensemble_models = []
    reconstructed_dataset = []
    for index, mapper in enumerate(feature_mapper):
        # extract dataset corresonding to mapper
        mapper_dataset = numpy_dataset[:, np.array(mapper)]
        mapper_dataset = np.reshape(mapper_dataset, (-1, 1, 1, len(mapper)))
        print(mapper_dataset.shape)

        # create models based on feature mapper
        model_input = keras.Input(
            shape=(1, 1, len(mapper)), name="input_".format(index)
        )
        model_seq = create_auto_encoder(
            len(mapper), 0.75, name="ae_ensemble_{}".format(index)
        )

        model_output = model_seq(model_input)

        model = keras.Model(model_input, model_output)

        model.summary()
        print(model.outputs)

        # train model
        logdir = "logs/ensemble_model_{}/".format(index) + datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=logdir, histogram_freq=1, write_images=True
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            # loss="binary_crossentropy",
            loss="mse",
            metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
        )

        model.fit(
            mapper_dataset,
            mapper_dataset,
            shuffle=True,
            epochs=n_epoches,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[tensorboard_callback,],
        )
        os.makedirs("models/ensemble_model_{0}/".format(index), exist_ok=True)
        model.save("models/ensemble_model_{0}/model_{0}.h5".format(index))

        reconstructed_dataset.append(model.predict(mapper_dataset))

        # predict on input dataset as input for output model
        # clean session for next training
        keras.backend.clear_session()

    # concat reconstructed outputs
    reconstructed_dataset = np.concatenate(reconstructed_dataset, axis=-1)
    print(reconstructed_dataset.shape)

    # create output model and train on reconstructed outputs
    output_model_len = sum(map(lambda x: len(x), feature_mapper))
    print("output_model_len:", output_model_len)

    # create models based on feature mapper
    out_model_input = keras.Input(shape=(1, 1, output_model_len), name="input_concat")

    out_model_seq = create_auto_encoder(output_model_len, 0.5, name="ae_output")

    model_output = out_model_seq(out_model_input)

    out_model = keras.Model(out_model_input, model_output)

    out_model.summary()
    print(out_model.outputs)

    # train model
    logdir = "logs/output_model/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=logdir, histogram_freq=1, write_images=True
    )

    out_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        # loss="binary_crossentropy",
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
    )

    out_model.fit(
        reconstructed_dataset,
        reconstructed_dataset,
        shuffle=True,
        epochs=n_epoches,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[tensorboard_callback,],
    )
    os.makedirs("models/output_model/", exist_ok=True)
    out_model.save("models/output_model/output_model.h5")

    keras.backend.clear_session()

    # tf_session = keras.backend.get_session()
    # # write out tensorflow checkpoint & meta graph
    # saver = tf.train.Saver()
    # save_path = saver.save(tf_session, "models/first_model/first_model.ckpt")

    # train each ensemble models and save to files
    # train output model and save to file
    # merge to single model file
