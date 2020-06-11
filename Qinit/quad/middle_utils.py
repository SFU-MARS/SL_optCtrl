import os,sys
import numpy as np
import pandas as pd
import pickle
from tensorflow import keras
import tensorflow as tf



class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

class NN_interp(object):
    def __init__(self, input_dim, hidden=None):
        self.model = keras.Sequential([
            keras.layers.Dense(hidden[0], activation=tf.nn.relu, input_shape=[input_dim]),
            keras.layers.Dense(hidden[1], activation=tf.nn.relu),
            keras.layers.Dense(1)
        ])
        self.optimizer = tf.keras.optimizers.Adadelta()
        self.model.compile(loss='mean_squared_error',
                      optimizer=self.optimizer,
                      metrics=['mean_absolute_error', 'mean_squared_error'])

    def train(self):
        # Configure all kinds of paths
        data_loadpath = "./train_data/valFunc_mpc_filled_final.csv"
        model_savepath = "./trained_model/nn_interp.h5"
        history_savepath = "./trained_model/nn_interp_history.csv"
        colnames = ['x', 'vx', 'z', 'vz', 'phi', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'reward', 'value', 'cost', 'collision_in_future', 'collision_current', 'col_trajectory_flag']
        
        # Prepare training data and label
        raw_dataset = pd.read_csv(data_loadpath, names=colnames, na_values="?", comment='\t', sep=",",
                                  skipinitialspace=True, skiprows=1)
        raw_dataset = raw_dataset.dropna()
        train_dataset = raw_dataset.sample(frac=1.0, random_state=0)
        train_labels = train_dataset.pop('value')

        # Pop some unneeded column names
        popped_colnames = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'reward', 'cost', 'collision_in_future', 'collision_current', 'col_trajectory_flag']
        for pop_name in popped_colnames:
            train_dataset.pop(pop_name)
        
        # Prepare model
        model = self.model
        model.summary()


        EPOCHS = 1000 
        history = model.fit(
            train_dataset, train_labels, batch_size=128,
            epochs=EPOCHS, validation_split=0.2, verbose=1,
            callbacks=[PrintDot()])

        keras.models.save_model(model, model_savepath)
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        hist.tail()
        hist.to_csv(history_savepath)

    def norm(self, x, stats):
        return (x - stats['mean']) / stats['std']

    def save_model_weights(self):
        model_savepath = "./trained_model/nn_interp.h5"
        dir_path = os.path.dirname(model_savepath)
        model_weights_savepath = dir_path + '/' + os.path.splitext(model_savepath.split('/')[-1])[0] + '_weights.pkl'
        assert os.path.exists(model_savepath)

        model = tf.keras.models.`load_model`(filepath=model_savepath)
        weights = []
        print("starting saving weights for model {}".format(model_savepath.split('/')[-1]))
        for layer in model.layers:
            weight = layer.get_weights()
            if layer.name.split('_')[0] != 'dropout':
                weights.append(weight)
                print("layer name:", layer.name)
                print("kernel weight shape of this layer:", np.shape(weight[0]))
                print("bias weight shape of this layer:", np.shape(weight[1]))
        print("weights:", weights)
        print("shape of weights:", np.shape(weights))

        with open(model_weights_savepath, 'wb') as f:
            pickle.dump(weights, f)
            print("saving weights successfully for model {}!!".format(model_savepath.split('/')[-1]))


if __name__ == "__main__":
    # Load raw MPC data file and use it to train an NN-based interpolator
    nn_interp = NN_interp(input_dim=6, hidden=[64,64])
    nn_interp.train()

    # Save the weight of trained NN-based interpolator
    nn_interp = NN_interp(input_dim=6, hidden=[64,64])
    nn_interp.save_model_weights()