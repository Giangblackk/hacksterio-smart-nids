import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import json
import tensorflow.compat.v1 as tf
from tensorflow import keras
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="merge mmodels into single one")
    parser.add_argument(
        "-i",
        "--trained-models-info",
        default="trained_models_info.json",
        help="Information of trained model in json format",
    )
    args = parser.parse_args()
    return args


def merge_models(trained_models_info_file):
    with open(trained_models_info_file, "r") as f:
        trained_models_info = json.load(f)

    ensemble_models = []
    ensemble_models_inputs = []
    ensemble_models_outputs = []
    for index, ensemble_model_path in enumerate(trained_models_info["ensembles"]):
        model = keras.models.load_model(ensemble_model_path)
        ensemble_models.append(model)
        ensemble_models_inputs.append(model.inputs[0])
        ensemble_models_outputs.append(model.outputs[0])

        print(index)
        print(model.inputs)
        print(model.outputs)

    concat_ensemble_output = keras.layers.concatenate(ensemble_models_outputs, axis=-1)

    output_model_path = trained_models_info["output"]
    output_model = keras.models.load_model(output_model_path)
    output_model.summary()
    output_model._layers.pop(0)  # remove input layer
    output_model.summary()
    output_model_seq = output_model.layers[0]  # get seq layer inside

    output_model_new_output = output_model_seq(
        concat_ensemble_output
    )  # apply on new input (concat ensemble output)

    total_model = keras.Model(ensemble_models_inputs, output_model_new_output)

    print(total_model.outputs)

    total_model.save("models/total2/total.h5")

    tf_session = keras.backend.get_session()
    # write out tensorflow checkpoint & meta graph
    saver = tf.train.Saver()
    save_path = saver.save(tf_session, "models/total2/total.ckpt")



if __name__ == "__main__":
    _args = parse_args()
    _trained_models_info_file = _args.trained_models_info
    merge_models(_trained_models_info_file)
