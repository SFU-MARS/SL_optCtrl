from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
import os
import pickle
import numpy as np
import pandas as pd

class MlpPolicy(object):
    recurrent = False

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope('vf'):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            print("obz:", obz)
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
                print("hid size:", hid_size)
            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        with tf.variable_scope('pol'):
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                # mean = tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name='final', kernel_initializer=U.normc_initializer(0.01))
                # logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                # logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.constant_initializer([-0.69, -1.6]))
                mean = tf.layers.dense(last_out, pdtype.param_shape()[0] // 2, name='final',
                                       kernel_initializer=U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0] // 2],
                                         initializer=tf.constant_initializer([-0.69]))
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                # pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final', kernel_initializer=U.normc_initializer(2.))
                mean = tf.layers.dense(last_out, pdtype.param_shape()[0] // 2, name='final',
                                       kernel_initializer=U.normc_initializer(0.01))

                logstd = tf.layers.dense(last_out, pdtype.param_shape()[0] // 2, name="logstd",
                                         kernel_initializer=U.normc_initializer(0.405))
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []

    def save_model(self, dirname, iteration=None):
        if iteration is not None and iteration != 'best':
            dirname = os.path.join(dirname, 'iter_%d' % iteration)
        elif iteration == 'best':
            dirname = os.path.join(dirname, 'best_model_so_far')
        else:
            dirname = os.path.join(dirname, 'trained_model')

        print('Saving model to %s' % dirname)
        U.save_state(dirname)
        print('Saved!')

    def load_model(self, dirname, iteration=None):
        if iteration is not None and iteration != 'best':
            dirname = os.path.join(dirname, 'iter_%d' % iteration)
        elif iteration == 'best':
            dirname = os.path.join(dirname, 'best_model_so_far')
        else:
            dirname = os.path.join(dirname, 'trained_model')
        
        print('Loading model from %s' % dirname)
        U.load_state(dirname)
        print('Loaded!')

    def print_model_details(self):
        U.display_var_info(self.get_trainable_variables())


class MlpPolicy_mod(object):
    recurrent = False

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True, load_weights_vf=False, load_weights_pol=False):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        # ob = tf.Print(ob, [ob])

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)
            # self.ob_rms.mean = tf.Print(self.ob_rms.mean, [self.ob_rms.mean], "xubo print ob_rms_mean:")
            # self.ob_rms.std  = tf.Print(self.ob_rms.std, [self.ob_rms.std], "xubo print ob_rms_std:")

        with tf.variable_scope('vf'):
            if not load_weights_vf:
                obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
                last_out = obz
                for i in range(num_hid_layers):
                    last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i" % (i + 1),
                                                          kernel_initializer=U.normc_initializer(1.0)))
                    print("hid size:", hid_size)
                self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:, 0]
            else:
                """
                    Loading value initialization computed by PPO itself, it's like re-using the pre-trained value function.
                """
                # print("we are loading value initialization computed by PPO itself !!!")
                # with open(os.environ['PROJ_HOME_3'] + "/tf_model/quad/vf_from_ppo_weights.pkl", 'rb') as f:
                #     weights = pickle.load(f)
                #     obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
                #     out_1 = tf.layers.dense(inputs=obz,
                #                             units=64,
                #                             name="fc1",
                #                             kernel_initializer=tf.constant_initializer(weights['pi/vf/fc1/kernel']),
                #                             bias_initializer=tf.constant_initializer(weights['pi/vf/fc1/bias']),
                #                             use_bias=True,
                #                             activation=tf.nn.tanh)
                #
                #     out_2 = tf.layers.dense(inputs=out_1,
                #                             units=64,
                #                             name="fc2",
                #                             kernel_initializer=tf.constant_initializer(weights['pi/vf/fc2/kernel']),
                #                             bias_initializer=tf.constant_initializer(weights['pi/vf/fc2/bias']),
                #                             use_bias=True,
                #                             activation=tf.nn.tanh)
                #
                #     self.vpred = tf.layers.dense(inputs=out_2,
                #                                  units=1,
                #                                  name="final",
                #                                  kernel_initializer=tf.constant_initializer(weights['pi/vf/final/kernel']),
                #                                  bias_initializer=tf.constant_initializer(weights['pi/vf/final/bias']),
                #                                  use_bias=True)[:,0]
                """
                    Loading external value initialization computed by value iteration.
                """
                with open(os.environ['PROJ_HOME_3'] + "/tf_model/dubinsCar/vf_weights.pkl", 'rb') as f:
                    weights = pickle.load(f)
                    # print("shape of weights:", np.array(weights).shape)
                    # print("weights:", weights)
                    print("loading external value weights!")

                    # --- prepare stats for proper normalization --- #
                    print("preparing stats mean and std for normalization ...")
                    val_filled_path = os.environ['PROJ_HOME_3'] + "/data/dubinsCar/valFunc_filled_cleaned.csv"
                    assert os.path.exists(val_filled_path)
                    colnames = ['x', 'y', 'theta', 'value', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7','d8']
                    val_filled = pd.read_csv(val_filled_path, names=colnames, na_values="?", comment='\t', sep=",",
                                             skipinitialspace=True, skiprows=1)

                    stats_source = val_filled.copy()
                    stats_source.dropna()
                    stats_source.pop('value')
                    stats = stats_source.describe()
                    stats = stats.transpose()

                    # obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
                    obz = tf.clip_by_value((ob - np.array(stats['mean'])) / np.array(stats['std']), -5.0, 5.0)
                    # ----------------------------------------------- #


                    out_1 = tf.layers.dense(inputs=obz,
                                              units=64,
                                              name="fc1",
                                              kernel_initializer=tf.constant_initializer(weights[0][0]),
                                              bias_initializer=tf.constant_initializer(weights[0][1]),
                                              use_bias=True,
                                              activation=tf.nn.tanh)

                    out_2 = tf.layers.dense(inputs=out_1,
                                              units=64,
                                              name="fc2",
                                              kernel_initializer=tf.constant_initializer(weights[1][0]),
                                              bias_initializer=tf.constant_initializer(weights[1][1]),
                                              use_bias=True,
                                              activation=tf.nn.tanh)

                    self.vpred = tf.layers.dense(inputs=out_2,
                                            units=1,
                                            name="final",
                                            kernel_initializer=tf.constant_initializer(weights[2][0]),
                                            bias_initializer=tf.constant_initializer(weights[2][1]),
                                            use_bias=True)[:,0]
                    # self.vpred = tf.Print(self.vpred, [self.vpred], "xubo print vpred: ")

        with tf.variable_scope('pol'):
            if not load_weights_pol:
                last_out = obz
                for i in range(num_hid_layers):
                    last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i' % (i + 1),
                                                          kernel_initializer=U.normc_initializer(1.0)))
                if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                    # mean = tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name='final', kernel_initializer=U.normc_initializer(0.01))
                    # logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                    # logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.constant_initializer([-0.69, -1.6]))
                    mean = tf.layers.dense(last_out, pdtype.param_shape()[0] // 2, name='final',
                                           kernel_initializer=U.normc_initializer(0.01))
                    logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0] // 2],
                                             initializer=tf.constant_initializer([-0.69]))
                    pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
                else:
                    # pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final', kernel_initializer=U.normc_initializer(2.))
                    mean = tf.layers.dense(last_out, pdtype.param_shape()[0] // 2, name='final',
                                           kernel_initializer=U.normc_initializer(0.01))

                    logstd = tf.layers.dense(last_out, pdtype.param_shape()[0] // 2, name="logstd",
                                             kernel_initializer=U.normc_initializer(0.405))
                    pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                with open(os.environ['PROJ_HOME_3'] + "/tf_model/quad/pf_weights.pkl", 'rb') as f:
                    weights = pickle.load(f)
                    print("shape of weights:", np.array(weights).shape)
                    print("weights:", weights)
                    print("loading external pol weights")
                    # print("load old but good weights to test")

                    out_1 = tf.layers.dense(inputs=obz,
                                            units=64,
                                            name="fc1",
                                            kernel_initializer=tf.constant_initializer(weights[0][0]),
                                            bias_initializer=tf.constant_initializer(weights[0][1]),
                                            use_bias=True,
                                            activation=tf.nn.tanh)

                    out_2 = tf.layers.dense(inputs=out_1,
                                            units=64,
                                            name="fc2",
                                            kernel_initializer=tf.constant_initializer(weights[1][0]),
                                            bias_initializer=tf.constant_initializer(weights[1][1]),
                                            use_bias=True,
                                            activation=tf.nn.tanh)

                    if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                        mean = tf.layers.dense(out_2, pdtype.param_shape()[0] // 2, name='final',
                                               kernel_initializer=tf.constant_initializer(weights[2][0]),
                                               bias_initializer=tf.constant_initializer(weights[2][1]),
                                               use_bias=True)
                        logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0] // 2],
                                                 initializer=tf.constant_initializer([-0.69]))
                        pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
                    else:
                        raise ValueError("gaussian var should be fixed !!")



        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []

    def save_model(self, dirname, iteration=None):
        if iteration is not None and iteration != 'best':
            dirname = os.path.join(dirname, 'iter_%d' % iteration)
        elif iteration == 'best':
            dirname = os.path.join(dirname, 'best_model_so_far')
        else:
            dirname = os.path.join(dirname, 'trained_model')

        print('Saving model to %s' % dirname)
        U.save_state(dirname)
        print('Saved!')

    def load_model(self, dirname, iteration=None):
        if iteration is not None and iteration != 'best':
            dirname = os.path.join(dirname, 'iter_%d' % iteration)
        elif iteration == 'best':
            dirname = os.path.join(dirname, 'best_model_so_far')
        else:
            dirname = os.path.join(dirname, 'trained_model')

        print('Loading model from %s' % dirname)
        U.load_state(dirname)
        print('Loaded!')

    def print_model_details(self):
        U.display_var_info(self.get_trainable_variables())
