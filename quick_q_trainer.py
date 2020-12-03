import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.python.ops import math_ops
import numpy as np
import pickle
from utils import logger



class Trainer(object):
    def __init__(self):
        self.input_shape = 13
        self.hidden_shape = [400, 300]

    def build_value_model(self):
        model = keras.Sequential([
            keras.layers.Dense(self.hidden_shape[0], activation=tf.nn.tanh, input_shape=[self.input_shape]),
            keras.layers.Dense(self.hidden_shape[1], activation=tf.nn.tanh),
            keras.layers.Dense(1)
        ])
        # optimizer = tf.keras.optimizers.RMSprop(0.001)
        optimizer = tf.keras.optimizers.Adam(lr=0.01, epsilon=1e-8)
        # optimizer = tf.keras.optimizers.Adadelta()
        model.compile(loss='mean_squared_error',
                      optimizer = optimizer,
                      metrics=['mean_absolute_error', 'mean_squared_error'])
        return model

    def train_valFunc(self):
        dataset_path = "./data/dubins/polFunc_vi_filled_cleaned.csv"
        column_names = ['x', 'y', 'theta', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'vel', 'ang_vel', 'value']
        model_saving_path = "./tf_model/dubinsCar/ddpg_q_model.h5"

        raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=",", skipinitialspace=True, skiprows=1)

        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
        train_dataset = dataset.sample(frac=1.0, random_state=0)

        stats = train_dataset.describe()
        # print(stats)

        # quad with vi
        stats.pop("value")

        stats = stats.transpose()


        train_labels = train_dataset.pop('value')


        model = self.build_value_model()
        model.summary()
        # normed_train_data = self.norm(train_dataset, stats)
        normed_train_data = train_dataset

        EPOCHS = 1000
        history = model.fit(
            normed_train_data, train_labels, batch_size=64,
            epochs=EPOCHS, validation_split=0.2, verbose=1,
            callbacks=[PrintDot()])
            # callbacks=[PrintDot(), LrDecay()])

        keras.models.save_model(model, model_saving_path)
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        hist.tail()
        hist.to_csv(os.path.join(os.environ['PROJ_HOME_3'], 'tf_model', 'dubinsCar', 'value_SL_history.csv'))

        # self.plot_history(history)

    def norm(self, x, stats):
        return (x - stats['mean']) / stats['std']


    def plot_history(self, history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error')
        plt.plot(hist['epoch'], hist['mean_absolute_error'],
               label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
               label = 'Val Error')
        # plt.ylim([0,5])
        plt.legend()

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error')
        plt.plot(hist['epoch'], hist['mean_squared_error'],
               label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_squared_error'],
               label = 'Val Error')
        # plt.ylim([0,20])
        plt.legend()
        plt.show()

    def save_model_weights(self, model_path):
        dir_path = os.path.dirname(model_path)
        model_weights_savepath = model_path + '_weights.pkl'

        print(model_path)
        assert os.path.exists(model_path)
        # model = tf.keras.models.load_model(filepath=model_path,  custom_objects={'customerized_loss': customerized_loss})
        model = tf.keras.models.load_model(filepath=model_path)
        weights = []
        print("starting saving weights for model {}".format(model_path.split('/')[-1]))
        for layer in model.layers:
            print("________________________")
            print("layer name:", layer.name)
            weight = layer.get_weights()
            if layer.name.split('_')[0] != 'dropout':
                weights.append(weight)
            print("________________________")
        print("weights:", weights)
        print("shape of weights:", np.shape(weights))

        with open(model_weights_savepath, 'wb') as f:
            pickle.dump(weights, f)
            print("saving weights successfully for model {}!!".format(model_path.split('/')[-1]))

    def test(self):
        model_path = "./tf_model/dubinsCar/ddpg_q_model.h5"
        model = tf.keras.models.load_model(filepath = model_path)

        dataset_path = "./data/dubins/polFunc_vi_filled_cleaned.csv"
        column_names = ['x', 'y', 'theta', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'vel', 'ang_vel', 'value']
        raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=",", skipinitialspace=True, skiprows=1)
        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
        train_dataset = dataset.sample(frac=1.0, random_state=0)
        stats = train_dataset.describe()
        mean = stats.loc[['mean']].to_numpy()
        std = stats.loc[['std']].to_numpy()
        mean = mean.reshape(-1)
        std = std.reshape(-1)
        state_mean = mean[:11]
        state_std = std[:11]
        action_mean = mean[11:13]
        action_std = std[11:13]
        # print(mean)
        # print(std)

        # print(state_mean)
        # print(state_std)
        # print(action_mean)
        # print(action_std)

        state = [-4.2386, -3.6900,  0.1317,  1.1328,  1.4048,  2.3014,  5.2067,  5.4090, 8.3708,  6.0052,  8.6398]
        # state = (state - state_mean) / state_std
        state = np.array(state)
        action = [-2.0000,  1.9138]
        # action = (action - action_mean) / action_std
        action = np.array(action)
        x = np.concatenate((state, action), axis = None)
        x = [-4.2386, -3.6900,  0.1317, 1.1328,  1.4048,  2.3014,  5.2067,  5.4090, 8.3708,  6.0052,  8.6398, -2.0000,  1.9138]
        x = np.array([x])
        # print(state)
        # print(action)
        print(x)
        print(model.predict(x))
        



class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')




if __name__ == "__main__":
    trainer = Trainer()
    # trainer.train_valFunc()
    # trainer.save_model_weights("./tf_model/dubinsCar/ddpg_q_model.h5")
    trainer.test()