import sys,os
import numpy as np

import seaborn as sns
import pickle

import pandas as pd

import tensorflow as tf
from gym import spaces

if __name__ == "__main__":
    """
        Loading value initialization computed by PPO itself, it's like re-using the pre-trained value function.
    """
    dir = os.environ['PROJ_HOME_3'] + "/runs_log/21-Nov-2019_02-00-17DubinsCarEnv-v0_hand_craft_ppo/"
    with open(dir + "vf_from_ppo_weights.pkl", 'rb') as model_f, open(dir + "ppo_valpred_itself.csv", 'r') as data_f:
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
        normed_data = normed_data.drop(normed_data.columns[-1], axis=1)

        normed_data = np.array(normed_data)


        # define tensor graph
        import baselines.common.tf_util as U
        ob_space = spaces.Box(low=np.array([0]*11), high=np.array([1]*11))
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
                                 use_bias=True)[:,0]


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
            res = sess.run([vpred], feed_dict={obz:normed_data})
            print(np.max(res[0]))
            print(np.min(res[0]))
            print(np.shape(res[0]))


        d = {'x':data['x'], 'y':data['y'], 'theta':data['theta'], 'well_trained_val_ppo':res[0]}
        new_df = pd.DataFrame(data=d)

        sns.scatterplot(x=new_df['x'], y=new_df['y'], hue=new_df['well_trained_val_ppo'])
        # res_for_heatmap = new_df.pivot(index='y', columns='x', values='well_trained_val_ppo')
        # sns.heatmap(res_for_heatmap, annot=True, fmt="g", cmap='viridis')
        plt.show()
        pass


            # vpred = tf.Print(vpred, [vpred], "xubo print vpred: ")


            # new_saver = tf.train.import_meta_graph(dir + 'model'+ '/best_model_so_far.meta')
            # new_saver.restore(sess, tf.train.latest_checkpoint(dir + 'model'))
            #
            # graph = tf.get_default_graph()
            # variables = tf.global_variables()
            # # print("variables:", variables)
            # # print("ops:", graph.get_operations())
            #
            # names = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
            # for na in names:
            #     print(na)
            # # print("names:", names)
            # # input = graph.get_tensor_by_name("Xinput:0")
