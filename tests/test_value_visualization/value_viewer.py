import tensorflow as tf
import os
import multiprocessing
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.python import pywrap_tensorflow
import pickle
import numpy as np
import pandas as pd
from gym import spaces
import shutil
import seaborn as sns

from keras.models import load_model
import matplotlib.pyplot as plt

def Extract_model_weights(loadir):
    with tf.device('/gpu:1'):

        """ Specify model loading path """
        # dirname = os.environ['PROJ_HOME_3'] + '/runs_log/02-Nov-2019_22-34-26PlanarQuadEnv-v0_hand_craft_ppo'
        # dirname = os.environ['PROJ_HOME_3'] + '/runs_log/21-Nov-2019_02-00-17DubinsCarEnv-v0_hand_craft_ppo'
        # dirname = os.environ['PROJ_HOME_3'] + '/runs_log/21-Nov-2019_20-39-09DubinsCarEnv-v0_hand_craft_ppo'
        # dirname = os.environ['PROJ_HOME_3'] + '/runs_log/21-Nov-2019_21-43-59DubinsCarEnv-v0_hand_craft_ppo'
        # dirname = os.environ['PROJ_HOME_3'] + '/runs_log_tests/26-Nov-2019_16-52-50DubinsCarEnv-v0_hand_craft_ppo_vf'

        """ Specify tf session configuration"""
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
        config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu)
        config.gpu_options.allow_growth = True

        """ Start session and load the best model trained """
        sess = tf.Session(config=config)
        # First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph(loadir + '/model/trained_model.meta')
        latest_ckp = tf.train.latest_checkpoint(loadir + '/model')
        saver.restore(sess, latest_ckp)

        """ Extract weights accordingly and save to .pkl file"""
        weights_dict = {}
        # print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='')
        reader = pywrap_tensorflow.NewCheckpointReader(latest_ckp)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in sorted(var_to_shape_map):
            # print("tensor_name: ", key)
            # print(reader.get_tensor(key))
            weights_dict[key] = reader.get_tensor(key)

        for key, val in weights_dict.items():
            print(key, ':', val)

        # savename = "./quad_baseline_model_weights.pkl"
        with open("./model_weights.pkl", 'wb') as f:
            pickle.dump(weights_dict, f)

        import glob
        res = glob.glob(loadir + '/ppo_valpred_*.csv')
        shutil.copy(res[0], './src_data.csv')

def Value_distribution_viz(agent):
    with open("./model_weights.pkl", 'rb') as model_f, open("./src_data.csv", 'r') as data_f:
        weights = pickle.load(model_f)

        data = pd.read_csv(data_f)
        # drop the colunm of additional index
        data = data.drop(data.columns[0], axis=1)
        # drop some middle rows with header names
        data = data.drop(data[data.x == 'x'].index)
        data = data.apply(pd.to_numeric, errors='coerce')

        stats = data.describe()
        stats = stats.transpose()

        print("stats mean:", stats['mean'])
        print("stats std:", stats['std'])
        normed_data = (data - stats['mean']) / stats['std']
        normed_data = np.clip(normed_data, -5, 5)
        if agent == 'dubinsCar':
            normed_data = normed_data.drop(normed_data.columns[-1], axis=1)
        elif agent == 'quad':
            normed_data = normed_data.drop(normed_data.columns[14:], axis=1)
        else:
            raise ValueError("invalid agent!")
        normed_data = np.array(normed_data)

        # define tensor graph
        import sys
        sys.path.append(os.environ["PROJ_HOME_3"])
        import baselines.common.tf_util as U

        ob_space = None
        if agent == 'quad':
            ob_space = spaces.Box(low=np.array([0] * 14), high=np.array([1] * 14))
        elif agent == "dubinsCar":
            ob_space = spaces.Box(low=np.array([0] * 11), high=np.array([1] * 11))
        else:
            raise ValueError("invalid agent!")
        sequence_length = None
        obz = U.get_placeholder(name="obz", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        out_1 = tf.layers.dense(inputs=obz,
                                units=64,
                                name="fc1",
                                kernel_initializer=tf.constant_initializer(weights['pi/vf/fc1/kernel']),
                                bias_initializer=tf.constant_initializer(weights['pi/vf/fc1/bias']),
                                use_bias=True,
                                activation=tf.nn.tanh)

        out_2 = tf.layers.dense(inputs=out_1,
                                units=64,
                                name="fc2",
                                kernel_initializer=tf.constant_initializer(weights['pi/vf/fc2/kernel']),
                                bias_initializer=tf.constant_initializer(weights['pi/vf/fc2/bias']),
                                use_bias=True,
                                activation=tf.nn.tanh)

        vpred = tf.layers.dense(inputs=out_2,
                                units=1,
                                name="final",
                                kernel_initializer=tf.constant_initializer(weights['pi/vf/final/kernel']),
                                bias_initializer=tf.constant_initializer(weights['pi/vf/final/bias']),
                                use_bias=True)[:, 0]

        import multiprocessing
        import matplotlib.pyplot as plt

        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
        config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()

            sess.run(init)
            res = sess.run([vpred], feed_dict={obz: normed_data})
            print(np.max(res[0]))
            print(np.min(res[0]))
            print(np.shape(res[0]))
        d = None
        if agent == 'dubinsCar':
            d = {'x': data['x'], 'y': data['y'], 'theta': data['theta'], 'well_trained_val_ppo': res[0]}
            new_df = pd.DataFrame(data=d)
            sns.scatterplot(x='x', y='y', data=new_df, hue='well_trained_val_ppo')
        elif agent == "quad":
            d = {'x': data['x'], 'vx': data['vx'], 'z':data['z'], 'vz':data['vz'], 'phi': data['phi'], 'w':data['w'], 'well_trained_val_ppo': res[0]}
            new_df = pd.DataFrame(data=d)
            sns.scatterplot(x='x', y='z', data=new_df, hue='well_trained_val_ppo')
        else:
            raise ValueError("invalid agent!")


        # sns.scatterplot(x=new_df['x'], y=new_df['y'], hue=new_df['well_trained_val_ppo'], markers=["x"])
        # res_for_heatmap = new_df.pivot(index='y', columns='x', values='well_trained_val_ppo')
        # sns.heatmap(res_for_heatmap, annot=True, fmt="g", cmap='viridis')

        plt.savefig("./vf_dist_viz.png")
        plt.show()

if __name__ == "__main__":
    # Dir = os.environ['PROJ_HOME_3'] + '/runs_log_tests/quad_task_exploration/good baseline/23-Feb-2020_08-40-05PlanarQuadEnv-v0_hand_craft_ppo'
    # Extract_model_weights(loadir=Dir)
    # Value_distribution_viz(agent='quad')


    # # loaded model path; real test data path; and normalization-base data path
    # vf_model = load_model(os.environ['PROJ_HOME_3'] + '/tf_model/quad/old_model_old_reward_trained_valNN/vf_mpc.h5')
    # realdata_path = os.environ['PROJ_HOME_3'] + '/tests/test_value_visualization/src_data.csv'
    # normbasedata_path = os.environ['PROJ_HOME_3'] + '/data/quad/old_model_old_reward_training_data/valFunc_mpc_filled_final.csv'
    #
    # # construct normalization base data
    # normbase_colnames = ['x', 'vx', 'z', 'vz', 'phi', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'reward', 'value',
    #             'cost', 'collision_in_future', 'collision_current', 'col_trajectory_flag']
    # normbase_df = pd.read_csv(normbasedata_path, names=normbase_colnames, na_values="?", comment='\t', sep=",",
    #                          skipinitialspace=True, skiprows=1)
    # stats_source = normbase_df.copy()
    # stats_source.dropna()
    # print("stats:", stats_source)
    # stats = stats_source.describe()
    # stats.pop("reward")
    # stats.pop("value")
    # stats.pop("cost")
    # stats.pop("collision_in_future")
    # stats.pop("collision_current")
    # stats.pop("col_trajectory_flag")
    # stats = stats.transpose()
    #
    #
    # # real_colnames = ['x', 'vz', 'z', 'vz', 'phi', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8',
    # #             'ppo_valpred_itself', 'atarg', 'tdlamret', 'rews', 'events']
    #
    # real_df = pd.read_csv(realdata_path)
    # # drop the colunm of additional index
    # real_df = real_df.drop(real_df.columns[0], axis=1)
    # # drop some middle rows with header names
    # real_df = real_df.drop(real_df[real_df.x == 'x'].index)
    # real_df = real_df.apply(pd.to_numeric, errors='coerce')
    #
    # real_df.pop('ppo_valpred_itself')
    # real_df.pop('atarg')
    # real_df.pop('tdlamret')
    # real_df.pop('rews')
    # real_df.pop('events')
    #
    # print("real df:", real_df.head())
    #
    #
    # def norm(x, stats):
    #     return (x - stats['mean']) / stats['std']
    #
    # normed_df = norm(real_df, stats)
    #
    # # print(normed_df.head())
    #
    # valpred = np.empty(normed_df.shape[0], dtype=float)
    # idx = 0
    # for _, row in normed_df.iterrows():
    #     tmp = vf_model.predict(np.array(row).reshape(1, -1))[0][0]
    #     valpred[idx] = tmp
    #     idx += 1
    #
    # d = {'x': real_df['x'], 'vx': real_df['vx'], 'z': real_df['z'], 'vz': real_df['vz'], 'phi': real_df['phi'], 'w': real_df['w'],
    #      'well_trained_val_ppo': valpred}
    #
    # draw_df = pd.DataFrame(data=d)
    # ax = sns.scatterplot(x='x', y='z', data=draw_df, hue='well_trained_val_ppo')
    # plt.show()


    # Use 6D value iteration model and view the value distribution
    realdata_path = os.environ['PROJ_HOME_3'] + '/tests/test_value_visualization/src_data.csv'
    # read the csv file
    real_df = pd.read_csv(realdata_path)
    # drop the colunm of additional index
    real_df = real_df.drop(real_df.columns[0], axis=1)
    # drop some middle rows with header names
    real_df = real_df.drop(real_df[real_df.x == 'x'].index)
    real_df = real_df.apply(pd.to_numeric, errors='coerce')

    real_df.pop('ppo_valpred_itself')
    real_df.pop('atarg')
    real_df.pop('tdlamret')
    real_df.pop('rews')
    real_df.pop('events')
    real_df.pop('d1')
    real_df.pop('d2')
    real_df.pop('d3')
    real_df.pop('d4')
    real_df.pop('d5')
    real_df.pop('d6')
    real_df.pop('d7')
    real_df.pop('d8')

    print("real df:", real_df.head())
    cols = real_df.columns.tolist()
    print("real df cols: ", cols)



    from value_iteration.helper_function import *

    val_interp_f_quad = value_interpolation_function_quad()
    val_interp_f_quad.setup()

    # To be consisten with the input of val_interp_f_quad, the column order should be changed.
    # cols = [cols[0], cols[2], cols[1], cols[3], cols[4], cols[5]]
    # print("real df cols after:", cols)
    #
    # real_df = real_df[cols]
    # print("real df after:", real_df.head())

    valpred = val_interp_f_quad.interpolate_value(real_df)
    d = {'x': real_df['x'], 'vx': real_df['vx'], 'z': real_df['z'], 'vz': real_df['vz'], 'phi': real_df['phi'], 'w': real_df['w'],
         'val_from_6d_vi': valpred}

    draw_df = pd.DataFrame(data=d)
    ax = sns.scatterplot(x='x', y='z', data=draw_df, hue='val_from_6d_vi')
    plt.show()



