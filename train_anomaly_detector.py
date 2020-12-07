import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

import json
from datetime import datetime
import math
import numpy as np

import tensorflow.compat.v1 as tf
from tensorflow import keras


class EmsembleModel(keras.Model):
    def __init__(self, input_size, beta):
        super(EmsembleModel, self).__init__()
        self.input_size = input_size
        self.hidden_layer_size = math.ceil(beta * input_size)
        self.seq = keras.models.Sequential(
            [
                keras.layers.Conv2D(
                    filters=self.hidden_layer_size, kernel_size=1, activation="relu"
                ),
                keras.layers.Conv2D(
                    filters=self.input_size, kernel_size=1, activation="relu"
                ),
            ]
        )

    def call(self, x):
        return self.seq(x)


if __name__ == "__main__":
    batch_size = 100
    n_epoches = 100
    # load dataset
    dataset_file = "capEC2AMAZ-O4EL3NG-172.31.69.26a.pcap.h5"
    dataset = keras.utils.HDF5Matrix(dataset_file, "dataset")

    numpy_dataset = dataset.data[...]

    dataset_max = np.max(numpy_dataset, axis=0)
    dataset_min = np.min(numpy_dataset, axis=0)

    numpy_dataset = (numpy_dataset - dataset_min) / (1e-10 + dataset_max - dataset_min)

    mapper_file = "mapper.json"

    with open(mapper_file, "r") as f:
        mapper = json.load(f)
    print(mapper)

    # create models based on feature mapper
    first_mapper = mapper[0]
    first_mapper_dataset = numpy_dataset[:, np.array(first_mapper)]
    first_mapper_dataset = np.reshape(first_mapper_dataset, (-1, 1, 1, len(first_mapper)))

    print(first_mapper_dataset.shape)

    first_input = keras.Input(shape=(1, 1, len(first_mapper)))
    first_model = keras.Model(
        first_input, EmsembleModel(len(first_mapper), 0.75)(first_input)
    )
    first_model.summary()
    print(first_model.outputs)

    logdir = "logs/first_model/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    first_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        # loss='binary_crossentropy',
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError(name='my_rmse')],
    )

    first_model.fit(
        first_mapper_dataset,
        first_mapper_dataset,
        shuffle=True,
        epochs=n_epoches,
        batch_size=batch_size,
        callbacks=[tensorboard_callback],
    )

    # first_model.save("first.h5") # cannot save Model Subclasses to .h5
    tf_session = keras.backend.get_session()
    # write out tensorflow checkpoint & meta graph
    saver = tf.train.Saver()
    save_path = saver.save(tf_session, "first_model/first_model.ckpt")

    # train each ensemble models and save to files
    # train output model and save to file
    # merge to single model file
