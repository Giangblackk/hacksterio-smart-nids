import math
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow import keras
import json


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
    # load dataset
    dataset_file = "capEC2AMAZ-O4EL3NG-172.31.69.26a.pcap.h5"
    dataset = keras.utils.HDF5Matrix(dataset_file, "dataset")

    numpy_dataset = dataset.data[...]

    mapper_file = "mapper.json"

    with open(mapper_file, "r") as f:
        mapper = json.load(f)
    print(mapper)

    first_mapper = mapper[0]
    first_mapper_dataset = numpy_dataset[:, np.array(first_mapper)]
    print(first_mapper_dataset.shape)
    first_input = keras.Input(shape=(1, 1, len(first_mapper)))
    first_model = keras.Model(first_input, EmsembleModel(len(first_mapper), 0.75)(first_input))
    print(first_model.summary())
    print(first_model.outputs)

    # first_model.save("first.h5") # cannot save Model Subclasses to .h5

    tf_session = keras.backend.get_session()

    # write out tensorflow checkpoint & meta graph
    saver = tf.train.Saver()
    save_path = saver.save(tf_session,"first_model/first_model.ckpt")

    # freeze checkpoint and meta graph to protobuf

    # create models based on feature mapper
    # train each ensemble models and save to files
    # train output model and save to file
    # merge to single model file

