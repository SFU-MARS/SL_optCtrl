import pandas as pd
import os

import seaborn as sns
print(sns.__version__)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.python.ops import math_ops
import numpy as np
import pickle

from baselines import logger


def customerized_loss(y_true, y_pred):
    return K.mean(math_ops.square(y_pred - y_true), axis=-1)

class Trainer(object):
    def __init__(self, method='vi', target='valFunc', agent='quad'):
        self.agent = agent
        self.method = method
        if self.agent == 'quad':
            if target == 'valFunc':
                self.input_shape = 14
                self.model = self.build_value_model(self.input_shape)
            elif target == 'polFunc':
                self.input_shape = 14
                self.model = self.build_policy_model(self.input_shape)
        elif self.agent == 'car':
            if target == 'valFunc':
                self.input_shape = 13
                self.model = self.build_value_model(self.input_shape)
            elif target == 'polFunc':
                self.input_shape = 13
                self.model = self.build_policy_model(self.input_shape)
        elif self.agent == 'dubinsCar':
            if target == 'valFunc':
                self.input_shape = 11
                self.model = self.build_value_model(self.input_shape)
            elif target == 'polFunc':
                self.input_shape = 11
                self.model = self.build_policy_model(self.input_shape)
        else:
            raise ValueError("invalid agent type")

    def build_value_model(self, input_shape):
        model = keras.Sequential([
            keras.layers.Dense(64, activation=tf.nn.tanh, input_shape=[input_shape]),
            keras.layers.Dense(64, activation=tf.nn.tanh),
            keras.layers.Dense(1)
        ])
        # optimizer = tf.keras.optimizers.RMSprop(0.001)
        # optimizer = tf.keras.optimizers.Adam(lr=0.01, epsilon=1e-8)
        optimizer = tf.keras.optimizers.Adadelta()
        model.compile(loss='mean_squared_error',
                      optimizer = optimizer,
                      metrics=['mean_absolute_error', 'mean_squared_error'])
        return model

    def build_policy_model(self, input_shape):
        model = keras.Sequential([
            keras.layers.Dense(64, activation=tf.nn.tanh, input_shape=[input_shape]),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation=tf.nn.tanh),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(2)
        ])
        optimizer = tf.keras.optimizers.RMSprop(0.001)
        model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=['mean_absolute_error', 'mean_squared_error'])
        return model


    def train_valFunc(self):
        # dirpath = os.path.dirname(__file__)
        # print(dirpath)
        dirpath = os.environ['PROJ_HOME_3']
        dataset_path = None
        column_names = None
        model_saving_path = None
        if self.agent == 'car':
            if not os.path.exists(dirpath + "/data/car/valFunc_filled.csv"):
                raise ValueError("can not find the training file for car example!!")
            else:
                dataset_path = dirpath + "/data/car/valFunc_filled.csv"
                column_names = ['x', 'y', 'theta', 'delta', 'vel', 'value', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
                model_saving_path = './tf_model/car/vf.h5'

        elif self.agent == 'quad':
            if not os.path.exists(dirpath + "/data/quad/valFunc_mpc_filled_final.csv"):
                raise ValueError("can not find the training file for quad example!!")
            else:
                dataset_path = dirpath + "/data/quad/valFunc_mpc_filled_final.csv"
                column_names = ['x', 'vx', 'z', 'vz', 'phi', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8',
                                'reward', 'value', 'cost', 'collision_in_future', 'collision_current', 'col_trajectory_flag']
                model_saving_path = './tf_model/quad/vf_mpc.h5'
                logger.log("We use training data from {}".format(dataset_path))


        elif self.agent == 'dubinsCar':
            if not os.path.exists(dirpath + "/data/dubinsCar/env_difficult/valFunc_filled_cleaned.csv") or \
                    not os.path.exists(dirpath + "/data/dubinsCar/env_difficult/valFunc_mpc_filled_cleaned.csv"):
                raise ValueError("can not find the training file for dubins car example!!")
            else:
                if self.method == 'vi':
                    dataset_path = dirpath + "/data/dubinsCar/env_difficult/valFunc_filled_cleaned.csv"
                    model_saving_path = './tf_model/dubinsCar/vf.h5'
                    column_names = ['x', 'y', 'theta', 'value', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
                elif self.method == 'mpc':
                    # print("we are using mpc cost data to train!")
                    # dataset_path = dirpath + "/data/dubinsCar/env_difficult/valFunc_mpc_filled_cleaned.csv"
                    # model_saving_path = './tf_model/dubinsCar/vf_mpc.h5'
                    # column_names = ['reward', 'value', 'cost', 'status', 'x', 'y', 'theta', 'd1', 'd2', 'd3', 'd4',
                    #                 'd5', 'd6', 'd7', 'd8']

                    print("we are using mpc cost data (with soft constaints) to train!")
                    dataset_path = dirpath + "/data/dubinsCar/env_difficult/valFunc_mpc_filled_cleaned_soft.csv"
                    model_saving_path = './tf_model/dubinsCar/vf_mpc_soft.h5'
                    column_names = ['x', 'y', 'theta', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'reward','value','cost','collision_in_future','collision_current','col_trajectory_flag']
        else:
            raise ValueError("invalid agent!!!")

        print(dataset_path)
        raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=",", skipinitialspace=True, skiprows=1)

        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
        train_dataset = dataset.sample(frac=1.0, random_state=0)

        stats = train_dataset.describe()
        stats.pop("reward")
        stats.pop("value")
        stats.pop("cost")
        stats.pop("collision_in_future")
        stats.pop("collision_current")
        stats.pop("col_trajectory_flag")

        # stats.pop("reward")
        # stats.pop("value")
        # stats.pop("cost")
        # stats.pop("status")
        stats = stats.transpose()

        train_dataset.pop('reward')
        train_dataset.pop('cost')
        train_dataset.pop('collision_in_future')
        train_dataset.pop('collision_current')
        train_dataset.pop('col_trajectory_flag')

        # train_dataset.pop('reward')
        # train_dataset.pop('cost')
        # train_dataset.pop('status')
        train_labels = train_dataset.pop('value')

        model = self.build_value_model(self.input_shape)
        model.summary()
        normed_train_data = self.norm(train_dataset, stats)

        EPOCHS = 2000
        history = model.fit(
            normed_train_data, train_labels, batch_size=64,
            epochs=EPOCHS, validation_split=0.2, verbose=1,
            callbacks=[PrintDot()])
            # callbacks=[PrintDot(), LrDecay()])

        keras.models.save_model(model, model_saving_path)
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        hist.tail()
        hist.to_csv(os.path.join(os.environ['PROJ_HOME_3'], 'tf_model', 'quad', 'value_SL_history.csv'))

        # self.plot_history(history)


    # def train_polFunc(self, less_data=False):
    #     dirpath = os.path.dirname(__file__)
    #     dataset_path = None
    #     if self.agent == 'car':
    #         if not os.path.exists(dirpath + "/data/car/polFunc_filled.csv"):
    #             raise ValueError("can not find the training file!!")
    #         else:
    #             dataset_path = dirpath + "/data/car/polFunc_filled.csv"
    #         column_names = ['x', 'y', 'theta', 'delta', 'vel', 'acc', 'steer_rate', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7','d8']
    #         model_saving_path = './tf_model/car/pf.h5'
    #     elif self.agent == 'quad':
    #         if not os.path.exists(dirpath + "/data/quad/polFunc_filled.csv"):
    #             raise ValueError("can not find the training file!!")
    #         else:
    #             dataset_path = dirpath + "/data/quad/polFunc_filled.csv"
    #         column_names = ['x', 'vx', 'z', 'vz', 'phi', 'w', 'a1', 'a2', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
    #         model_saving_path = './tf_model/quad/pf.h5'
    #     raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=",",
    #                               skipinitialspace=True, skiprows=1)
    #
    #     dataset = raw_dataset.copy()
    #     print(dataset.tail())
    #     print(dataset.isna().sum())
    #
    #     dataset = dataset.dropna()
    #     if less_data:
    #         train_dataset_part1 = dataset.iloc[1:100:nrow(dataset), :]
    #         train_dataset_part2 = dataset.iloc[2:100:nrow(dataset), :]
    #         train_dataset = pd.concat(train_dataset_part1, train_dataset_part2)
    #     else:
    #         train_dataset = dataset.sample(frac=1.0, random_state=0)
    #
    #     if self.agent == 'quad':
    #         train_stats = train_dataset.describe()
    #         train_stats.pop("a1")
    #         train_stats.pop("a2")
    #         train_stats = train_stats.transpose()
    #         train_labels = pd.concat([train_dataset.pop(x) for x in ['a1', 'a2']], 1)
    #         # re-scale action for quad, from [0,12] -> [-1,1], then transform back at PlanarQuad env step function
    #         train_labels = -1 + (1 - (-1)) * (train_labels - 0) / (12 - 0)
    #     elif self.agent == 'car':
    #         train_stats = train_dataset.describe()
    #         train_stats.pop("acc")
    #         train_stats.pop("steer_rate")
    #         train_stats = train_stats.transpose()
    #         train_labels = pd.concat([train_dataset.pop(x) for x in ['acc', 'steer_rate']], 1)
    #
    #     model = self.build_policy_model(self.input_shape)
    #     model.summary()
    #     normed_train_data = self.norm(train_dataset, train_stats)
    #
    #     EPOCHS = 2500
    #
    #     history = model.fit(
    #         normed_train_data, train_labels,
    #         epochs=EPOCHS, validation_split=0.2, verbose=1,
    #         callbacks=[PrintDot()])
    #
    #     keras.models.save_model(model, model_saving_path)
    #     hist = pd.DataFrame(history.history)
    #     hist['epoch'] = history.epoch
    #     hist.tail()
    #
    #     # self.plot_history(history)
    #
    # def train_valFunc_merged(self):
    #     dirpath = os.path.dirname(__file__)
    #     val_filled_path = None
    #     val_filled_mpc_path = None
    #     if self.agent == 'car':
    #         val_filled_mpc_path = dirpath + "/data/car/valFunc_mpc_filled.csv"
    #         val_filled_path = dirpath + "/data/car/valFunc_filled.csv"
    #         assert os.path.exists(val_filled_mpc_path) and os.path.exists(val_filled_path)
    #         column_names = ['x', 'y', 'theta', 'delta', 'vel', 'value', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
    #         model_saving_path = './tf_model/car/vf_merged.h5'
    #     elif self.agent == 'quad':
    #         val_filled_mpc_path = dirpath + "/data/quad/valFunc_mpc_filled.csv"
    #         val_filled_path = dirpath + "/data/quad/valFunc_filled.csv"
    #         assert os.path.exists(val_filled_mpc_path) and os.path.exists(val_filled_path)
    #         column_names = ['x', 'vx', 'z', 'vz', 'phi', 'w', 'value', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
    #         model_saving_path = './tf_model/quad/vf_merged.h5'
    #
    #     val_filled = pd.read_csv(val_filled_path, names=column_names, na_values="?", comment='\t', sep=",", skipinitialspace=True, skiprows=1)
    #     val_filled_mpc = pd.read_csv(val_filled_mpc_path, names=column_names, na_values="?", comment='\t', sep=",", skipinitialspace=True, skiprows=1)
    #
    #     dataset = pd.concat([val_filled.copy(), val_filled_mpc.copy()])
    #     dataset = dataset.dropna()
    #     train_dataset = dataset.sample(frac=1.0, random_state=0)
    #
    #     stats = train_dataset.describe()
    #     stats.pop("value")
    #     stats = stats.transpose()
    #
    #     train_labels = train_dataset.pop('value')
    #
    #     model = self.build_value_model(self.input_shape)
    #     model.summary()
    #     normed_train_data = self.norm(train_dataset, stats)
    #
    #     EPOCHS = 2500
    #     history = model.fit(
    #         normed_train_data, train_labels, batch_size=128,
    #         epochs=EPOCHS, validation_split=0.2, verbose=1,
    #         callbacks=[PrintDot()])
    #
    #     keras.models.save_model(model, model_saving_path)
    #     hist = pd.DataFrame(history.history)
    #     hist['epoch'] = history.epoch
    #     hist.tail()
    #
    #     self.plot_history(history)


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
        model_weights_savepath = dir_path + '/' + os.path.splitext(model_path.split('/')[-1])[0] + '_weights.pkl'

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



class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

# class LrDecay(keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs):
#         lr = self.model.optimizer.lr
#         lr_with_decay = lr
#         if epoch % 500 == 0:
#             lr_with_decay = lr * 0.5
#             print("updating lr by 0.5!!!")
#         self.model.optimizer.lr = lr_with_decay
        # print(tf.Print(iterations, [iterations]))
        # print(K.eval(lr_with_decay))
        # self.model.optimizer.lr = lr_with_decay
        # print(K.eval(self.model.optimizer.lr))



if __name__ == "__main__":
    """
    Train policy network model and save weights
    """
    # trainer = Trainer(target="polFunc", agent='quad')
    # trainer.train_polFunc(less_data=False)
    # trainer.save_model_weights(type='pol')


    """
    Train value network model
    """
    # trainer = Trainer(method='mpc', target="valFunc", agent='dubinsCar')
    # trainer.train_valFunc()
    # trainer.save_model_weights("./tf_model/dubinsCar/vf_mpc_soft.h5")

    trainer = Trainer(method='mpc', target="valFunc", agent='quad')
    trainer.train_valFunc()
    trainer.save_model_weights("./tf_model/quad/vf_mpc.h5")

