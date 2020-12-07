import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import numpy as np
import json
import tensorflow.compat.v1 as tf
from tensorflow import keras

if __name__ == "__main__":
    # load model
    merged_model_path = "models/total2/total.h5"
    model = keras.models.load_model(merged_model_path)
    # load mapper
    mapper_file = "mapper.json"

    with open(mapper_file, "r") as f:
        feature_mapper = json.load(f)
    print(feature_mapper)
    # read input and transform for inference
    inputs = np.random.rand(1, 100).astype(np.float32)
    transformed_inputs = [
        np.reshape(inputs[:, np.array(m)], (-1, 1, 1, len(m))) for m in feature_mapper
    ]
    concat_transform_inputs = np.concatenate(transformed_inputs, axis=-1)
    # print(transformed_inputs)
    # inference
    predictions = model.predict(transformed_inputs)
    print(predictions.shape)
    # get RMSE
    RMSE = np.sqrt(np.mean((predictions - concat_transform_inputs) ** 2))
    print(RMSE)
