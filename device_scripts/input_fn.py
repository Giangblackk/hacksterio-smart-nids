import math
from itertools import cycle
import numpy as np
import json
import h5py

calib_batch_size = 512

mapping_file = "mapper.json"
dataset_norm_file = "norm.json"
dataset_file = "capEC2AMAZ-O4EL3NG-172.31.69.26a.pcap.h5"

with open(mapping_file, "r") as f:
    feature_mapper = json.load(f)
print(feature_mapper)

with open(dataset_norm_file, "r") as f:
    data_norm = json.load(f)

dataset_max = np.array(data_norm["max"], dtype=np.float32)
dataset_min = np.array(data_norm["min"], dtype=np.float32)

with h5py.File(dataset_file, "r") as f:
    dataset = f["dataset"][...]

norm_dataset = (dataset - dataset_min) / (1e-10 + dataset_max - dataset_min)

dataset_len = dataset.shape[0]
print(dataset.shape)

dataset_indice_pool = cycle(range(dataset_len)) # cirular iterator over dataset index


def calib_input(iter):
    selected_indices = [next(dataset_indice_pool) for _ in range(calib_batch_size)]
    selected_inputs = norm_dataset[np.array(selected_indices)]

    next_preprocessed_input = {}
    for index, mapper in enumerate(feature_mapper):
        input_name = "input{}".format(index)
        features = selected_inputs[:, np.array(mapper)]
        next_preprocessed_input[input_name] = np.reshape(features, (-1, 1, 1, len(mapper)))
    return next_preprocessed_input
