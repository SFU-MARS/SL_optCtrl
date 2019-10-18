import tensorflow as tf
from keras.models import load_model

import torch
import os

if __name__ == "__main__":
    model = load_model(os.environ['PROJ_HOME_3'] + '/tf_model/car/vf.h5')
    weights = model.get_weights()
    # print(weights)
    # print("weight shape:", len(weights))

    for i in range(len(weights)):
        print("weight %d:" %(i), weights[i])