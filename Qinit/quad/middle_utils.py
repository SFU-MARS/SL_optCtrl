import os,sys
import numpy as np
import pandas as pd
import pickle
from tensorflow import keras
import tensorflow as tf


def save_model_weights(model_savepath):
    dir_path = os.path.dirname(model_savepath)
    model_weights_savepath = dir_path + '/' + os.path.splitext(model_savepath.split('/')[-1])[0] + '_weights.pkl'
    assert os.path.exists(model_savepath)

    model = tf.keras.models.load_model(filepath=model_savepath)
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


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


# Re-organize the sensor-filled data from raw MPC data to include V(s), r(s) and Q(s,a)
class Qinit_DataGen(object):
    def __init__(self, agent):
        assert agent == "quad"
        self.agent = agent
        self.goal_state = np.array([4.0, 9.0, 0])  # for the newest data Seth just generates
        self.goal_tolerance = np.array([1.0, 1.0, np.pi/3])

        self.mpc_horizon = 140
        self.discount    = 0.9

        self.raw_mpc_path = "/local-scratch/xlv/SL_optCtrl/Qinit/quad/train_data/test_samps_800_N140_warmstart_big_region.csv"
        self.qfilled_mpc_path = "/local-scratch/xlv/SL_optCtrl/Qinit/quad/train_data/test_samps_800_N140_warmstart_big_region_qfilled.csv"
        
    def run(self):
        raw_mpc = np.genfromtxt(self.raw_mpc_path, delimiter=',', skip_header=True, dtype=np.float32)
        T = np.shape(raw_mpc)[0]

        rews = np.zeros(T, 'float32')
        vpreds = np.zeros(T, 'float32')
        qpreds = np.zeros(T, 'float32')

        for i in range(self.mpc_horizon, T + 1, self.mpc_horizon):
            for j in reversed(range(i - self.mpc_horizon, i)):
                if raw_mpc[j, -5]:  # column name: in target currently
                    rews[j] = 1000
                elif raw_mpc[j, -2]:
                    rews[j] = -400
                else:
                    rews[j] = 0

                if j == i - 1 or rews[j] == 1000 or rews[j] == -400:
                    vpreds[j] = rews[j]
                    qpreds[j] = rews[j]
                else:
                    vpreds[j] = rews[j] + self.discount * vpreds[j + 1]
                    qpreds[j] = rews[j] + self.discount * vpreds[j + 1]
        
        rews = rews.reshape(-1, 1)
        vpreds = vpreds.reshape(-1, 1)

        df = pd.read_csv(self.raw_mpc_path)
        df['rews'] = rews
        df['vpreds'] = vpreds
        df['qpreds'] = qpreds

        df.to_csv(self.qfilled_mpc_path)


# This is for TD3 on PlanarQuad example using raw MPC data
class NN_interp(object):
    def __init__(self, type, hidden=None):
        self.type = type
        if self.type == "vnn":
            self.input_dim = 6  # only includes 6d internal state
            self.model = keras.Sequential([
                keras.layers.Dense(hidden[0], activation=tf.nn.relu, input_shape=[self.input_dim]),
                keras.layers.Dense(hidden[1], activation=tf.nn.relu),
                keras.layers.Dense(1)])
        elif self.type == "qnn":
            self.input_dim = 8  # only includes 6d internal state and 2d action
            self.model = keras.Sequential([
                keras.layers.Dense(hidden[0], activation=tf.nn.relu, input_shape=[self.input_dim]),
                keras.layers.Dense(hidden[1], activation=tf.nn.relu),
                keras.layers.Dense(1)])
        else:
            raise ValueError("please choose a valid type for NN interp ...")
        self.optimizer = tf.keras.optimizers.Adadelta()
        self.model.compile(loss='mean_squared_error',
                      optimizer=self.optimizer,
                      metrics=['mean_absolute_error', 'mean_squared_error'])

    def train(self):
        # Configure all kinds of paths
        if self.type == "qnn":
            self.data_loadpath = "./train_data/test_samps_800_N140_warmstart_big_region_qfilled.csv"
            self.model_savepath = "./trained_model/qnn_interp/nn_interp.h5"
            self.history_savepath = "./trained_model/qnn_interp/nn_interp_history.csv"
            self.colnames = ['samp', 'x', 'vx', 'z', 'vz', 'phi', 'w', 'T1' ,'T2' ,'status', 'start_in_obstacle' ,'collision_in_trajectory', 'in_target_currently', 'end_in_target', 'collision_in_future', 'collision_current', 'col_trajectory_flag', 'rews', 'vpreds', 'qpreds']
        elif self.type == "vnn":
            self.data_loadpath = "./train_data/valFunc_mpc_filled_final.csv"
            self.model_savepath = "./trained_model/vnn_interp/nn_interp.h5"
            self.history_savepath = "./trained_model/vnn_interp/nn_interp_history.csv"
            self.colnames = ['x', 'vx', 'z', 'vz', 'phi', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'reward', 'value', 'cost', 'collision_in_future', 'collision_current', 'col_trajectory_flag']
        else:
            raise ValueError("invalid type")
        
        # Prepare training data and label
        raw_dataset = pd.read_csv(self.data_loadpath, names=self.colnames, na_values="?", comment='\t', sep=",", skipinitialspace=True, skiprows=1)
        raw_dataset = raw_dataset.dropna()
        train_dataset = raw_dataset.sample(frac=1.0, random_state=0)
        train_labels = train_dataset.pop('value') if self.type == "vnn" else train_dataset.pop('qpreds')

        # Pop some unneeded column names
        if self.type == "vnn":
            popped_colnames = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'reward', 'cost', 'collision_in_future', 'collision_current', 'col_trajectory_flag']
        elif self.type == "qnn":
            popped_colnames = ['samp', 'status', 'start_in_obstacle' ,'collision_in_trajectory', 'in_target_currently', 'end_in_target', 'collision_in_future', 'collision_current', 'col_trajectory_flag', 'rews', 'vpreds']
        else:
            raise ValueError("invalid type")
        for pop_name in popped_colnames:
            train_dataset.pop(pop_name)
        
        # Assert dataframe column names
        if self.type == "vnn":
            print(train_dataset.columns)
            print(train_dataset.head())
            print(train_labels.head())
            # assert train_dataset.columns == ['x', 'vx', 'z', 'vz', 'phi', 'w']
        elif self.type == "qnn":
             print(train_dataset.columns)
             print(train_dataset.head())
             print(train_labels.head())
            # assert train_dataset.columns == ['x', 'vx', 'z', 'vz', 'phi', 'w', 'T1' ,'T2']
        
        
        # Prepare model
        model = self.model
        model.summary()


        EPOCHS = 500
        history = model.fit(
            train_dataset, train_labels, batch_size=128,
            epochs=EPOCHS, validation_split=0.2, verbose=1,
            callbacks=[PrintDot()])

        keras.models.save_model(model, self.model_savepath)
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        hist.tail()
        hist.to_csv(self.history_savepath)

    def norm(self, x, stats):
        return (x - stats['mean']) / stats['std']

    


if __name__ == "__main__":
    # Load raw MPC data file and use it to train an NN-based interpolator
    nn_interp = NN_interp(type="qnn", hidden=[64,64])
    nn_interp.train()

    # Save the weight of trained NN-based interpolator
    save_model_weights("./trained_model/qnn_interp/nn_interp.h5")


# if __name__ == "__main__":
#     qdg = Qinit_DataGen('quad')
#     qdg.run()