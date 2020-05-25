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


class QNN(object):
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

    def train(self, data_loadpath, model_savepath, history_savepath, colnames):
        # Prepare training data and label
        raw_dataset = pd.read_csv(data_loadpath, names=colnames, na_values="?", comment='\t', sep=",",
                                  skipinitialspace=True, skiprows=1)
        raw_dataset = raw_dataset.dropna()
        train_dataset = raw_dataset.sample(frac=0.5, random_state=0)
        train_labels = train_dataset.pop('Q')
        # stats = train_dataset.describe()
        # stats = stats.transpose()
        # normed_train_data = self.norm(train_dataset, stats)

        # Prepare model
        model = self.model
        model.summary()


        EPOCHS = 400  # too many iterations cause over-fitting
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

    def save_model_weights(self, model_savepath):
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

def LoadValMatrix(datapath):
    return np.load(datapath)

def PrintNormInfo(datapath, colnames):
    df = pd.read_csv(datapath, names=colnames, na_values="?", comment='\t', sep=",", skipinitialspace=True, skiprows=1)
    df = df.dropna()

    df_stats = df.describe()
    df_stats = df_stats.transpose()
    return df_stats['mean'], df_stats['std']

if __name__ == "__main__":
    np.random.seed(0)
    tf.set_random_seed(1)

    from value_iteration.value_iteration_6d_xubo_version_1.value_iteration_6d_xubo_version_1_boltzmann import \
        env_quad_6d
    from value_iteration.value_iteration_3d.value_iteration_car_3d import env_dubin_car_3d
    # hidden = [400, 300]
    hidden = [256, 256]
    user_config = {'agent': 'dubinsCar',
                   'input_dim': 13,
                   'hidden': hidden,
                   'data_loadpath': "/local-scratch/xlv/SL_optCtrl/Qinit/dubinsCar/train_data/Qval.csv",
                   'model_savepath': "/local-scratch/xlv/SL_optCtrl/Qinit/dubinsCar/trained_model/{}*{}/Qf.h5".format(hidden[0], hidden[1]),
                   'history_savepath': "/local-scratch/xlv/SL_optCtrl/Qinit/dubinsCar/trained_model/{}*{}/Qf_history.csv".format(hidden[0], hidden[1]),
                   'column_names': ['x', 'y', 'theta', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'a1', 'a2', 'Q'],
                   'valM_loadpath': "/local-scratch/xlv/SL_optCtrl/value_iteration/value_boltzmann_angle.npy",
                   'env_id': "DubinsCarEnv-v0",
                   'rew_type': "hand_craft",
                   'set_additional_goal': "angle",
                   'gamma': 0.998}


    # # Load offline value matrix
    # valM = LoadValMatrix(datapath=user_config['valM_loadpath'])
    #
    # # Run one step simulation in Gazebo, using sampled action
    # Nsamps = 500000
    # idsamps = 0
    # env = gym.make(user_config['env_id'], reward_type=user_config['rew_type'], set_additional_goal=user_config['set_additional_goal'])
    # subenv = None
    #
    # if user_config['agent'] == 'dubinsCar':
    #     subenv = env_dubin_car_3d()
    # else:
    #     subenv = env_quad_6d()
    # subenv.algorithm_init()
    # interp = subenv.set_interpolation(valM)
    #
    # # Prepare write to csv file and serve as Q func training data.
    # with open(user_config['data_loadpath'], 'w', newline='') as Qfile:
    #     writer = csv.DictWriter(Qfile, user_config['column_names'])
    #     writer.writeheader()
    #
    #     while idsamps < Nsamps:
    #         # subenv.reset makes sure that s is non-collision state (3-dim)
    #         s = subenv.reset()
    #         o = env.reset(spec=s)
    #         a = env.action_space.sample()
    #         o_prime, r, d, info = env.step(a)
    #         if np.isnan(o_prime).any() or np.isinf(o_prime).any():
    #             continue
    #         s_prime = o_prime[:3] if user_config['agent'] == "dubinsCar" else o_prime[:6]
    #
    #         val_s = interp(s)
    #         val_s_prime = interp(s_prime)
    #
    #         Q_sa = r + user_config['gamma'] * val_s_prime
    #
    #         # Save the Q value for current s and a to offline csv file
    #         tmp_keys = user_config['column_names']
    #         tmp_values = o.tolist() + a.tolist() + Q_sa.tolist()
    #         # print("tmp values:", tmp_values)
    #         tmp_dict = dict(zip(tmp_keys, tmp_values))
    #         writer.writerow(tmp_dict)
    #
    #         idsamps += 1
    #         if idsamps % 1000 == 0:
    #             print("processing {} rows".format(idsamps))
    #
    # # Clear Q table file
    # Qdf = pd.read_csv(user_config['data_loadpath'])
    # Qdf = Qdf.replace([np.inf, -np.inf], np.nan)
    # Qdf = Qdf.dropna()
    # Qdf.to_csv(user_config['data_loadpath'])

    # Load this Q data file and use it to train the Q network
    q_net = QNN(input_dim=user_config['input_dim'], hidden=user_config['hidden'])
    q_net.train(data_loadpath=user_config['data_loadpath'],
                model_savepath=user_config['model_savepath'],
                history_savepath=user_config['history_savepath'],
                colnames=user_config['column_names'])

    # Save the weight of trained Q model
    q_net = QNN(input_dim=user_config['input_dim'], hidden=user_config['hidden'])
    q_net.save_model_weights(user_config['model_savepath'])


    # mean, std = PrintNormInfo(user_config['data_loadpath'], user_config['column_names'])
    # print("mean:", mean, 'std:', std)