import tensorflow as tf
from keras.models import load_model

import torch
import os

import baselines.common.tf_util as U

# if __name__ == "__main__":
#     model = load_model(os.environ['PROJ_HOME_3'] + '/tf_model/car/vf.h5')
#     weights = model.get_weights()
#     # print(weights)
#     # print("weight shape:", len(weights))
#
#     for i in range(len(weights)):
#         print("weight %d:" %(i), weights[i])


if __name__ == "__main__":
    import multiprocessing

    from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
    from tensorflow.python import pywrap_tensorflow
    import pickle

    with tf.device('/gpu:1'):
        dirname = os.environ['PROJ_HOME_3'] + '/runs_log/02-Nov-2019_22-34-26PlanarQuadEnv-v0_hand_craft_ppo/model'
        num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
        config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu)
        config.gpu_options.allow_growth = True

        sess = tf.Session(config=config)
        # First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph(dirname + '/best_model_so_far.meta')
        latest_ckp = tf.train.latest_checkpoint(dirname)
        saver.restore(sess, latest_ckp)


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

        with open(os.environ['PROJ_HOME_3']+'/tf_model/quad/vf_from_ppo_weights.pkl', 'wb') as f:
            pickle.dump(weights_dict, f)