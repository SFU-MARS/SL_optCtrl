import gym
import tensorflow as tf
import numpy as np
import pickle

from ppo1 import ppo
from utils import logger
from utils.tools import *
from gym_foo import gym_foo


from time import *
import argparse
import json


def run(env, algorithm, args, params=None, load=False, loadpath=None, loaditer=None, save_obs=False):

    assert algorithm == ppo
    assert args['gym_env'] in ["PlanarQuadEnv-v0", "DubinsCarEnv-v0", "DubinsCarEnv-v1"]

    # Initialize policy
    ppo.create_session()
    init_policy = ppo.create_policy('pi', env, args=args, vf_load=True if args['vf_load'] == "yes" else False, pol_load=True if args['pol_load'] == "yes" else False)
    ppo.initialize()


    # load trained policy
    if load and loadpath is not None and loaditer is not None:
        print("^^^^^^^^^^^^^^^")
        pi = init_policy
        pi.load_model(loadpath, iteration=loaditer)
        pi.save_model(args['MODEL_DIR'], iteration=0)
    else:
        # init policy
        pi = init_policy
        pi.save_model(args['MODEL_DIR'], iteration=0)

    # init params
    with open(params) as params_file:
        d = json.load(params_file)
        # num_iters = d.get('num_iters')  // no longer required to avoid possible performance peak from re-loading model

        # num_ppo_iters = d.get('num_ppo_iters')
        # optim_epochs = d.get('optim_epochs')
        # optim_stepsize = d.get('optim_stepsize')
        # gamma = d.get('gamma')


        num_ppo_iters = args['num_ppo_iters']
        optim_epochs = args['optim_epochs']
        optim_stepsize = args['optim_stepsize']
        timesteps_per_actorbatch = args['timesteps_per_actorbatch']
        gamma = args['gamma']


        # timesteps_per_actorbatch = d.get('timesteps_per_actorbatch')
        clip_param = d.get('clip_param')
        entcoeff = d.get('entcoeff')
        optim_batchsize = d.get('optim_batchsize')
        lam = d.get('lam')
        max_iters = num_ppo_iters

        lam = args['lam']
        logger.log("running lam is:", lam)
        logger.log("optim_stepsize: ", optim_stepsize)
        logger.log("optim_epochs: ", optim_epochs)
        logger.log("num_ppo_iters: ", num_ppo_iters)
        logger.log("kl threshold: ", args['kl'])


    if args['run_type'] == "train":
        pi = algorithm.ppo_learn(env=env, policy=pi, timesteps_per_actorbatch=timesteps_per_actorbatch,
                                 clip_param=clip_param, entcoeff=entcoeff, optim_epochs=optim_epochs,
                                 optim_stepsize=optim_stepsize, optim_batchsize=optim_batchsize,
                                 gamma=gamma, lam=lam,  args=args, max_iters=max_iters, schedule='constant', save_obs=save_obs)
    elif args['run_type'] == "eval":
        pi = algorithm.ppo_eval(env=env, policy=pi, timesteps_per_actorbatch=timesteps_per_actorbatch,max_iters=5)
    else:
        logger.log("invalid run_type!")
    env.close()

    return pi

if __name__ == "__main__":
    with tf.device('/gpu:1'):
        # ----- path setting ------
        parser = argparse.ArgumentParser()
        parser.add_argument("--gym_env", help="which gym environment to use.", type=str, default='PlanarQuadEnv-v0')
        parser.add_argument("--reward_type", help="which type of reward to use.", type=str, default='hand_craft')
        parser.add_argument("--algo", help="which type of algorithm to use.", type=str, default='ppo')
        parser.add_argument("--run_type", help="train or eval", type=str, default="train")
        parser.add_argument("--set_additional_goal", type=str, default="angle")
        parser.add_argument("--vf_load", help="yes or no", type=str, default="")
        parser.add_argument("--pol_load", help="yes or no", type=str, default="")
        parser.add_argument("--vf_type", help="vi_gd, boltzmann or mpc, if vf_load is yes", type=str, default="")
        parser.add_argument("--vf_switch", help="yes, no or always", type=str, default="")

        parser.add_argument("--lam", type=float, default=1.0)
        parser.add_argument("--grad_norm", type=float, default=0.5)
        parser.add_argument("--optim_stepsize", type=float, default=3e-4)
        parser.add_argument("--num_ppo_iters", type=int, default=300)
        parser.add_argument("--optim_epochs", type=int, default=10)
        parser.add_argument("--timesteps_per_actorbatch", type=int, default=256)
        parser.add_argument("--kl", type=float, default=0.5)
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--difficulty", type=str, default="hard")


        parser.add_argument("--adv_shift", type=str, default="no")
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--method", type=str, default="ppo")
        parser.add_argument("--policy_iteration", type=int, default=0)

        args = parser.parse_args()
        args = vars(args)

        print(args['method'])

        RUN_DIR = MODEL_DIR = FIGURE_DIR = RESULT_DIR = None
        if args['algo'] == "ppo":
            RUN_DIR = os.path.join(os.getcwd(), 'runs_log_tests',
                                   strftime('%d-%b-%Y_%H-%M-%S') + args['gym_env'] + '_' + args['reward_type'] + '_' + args['algo'])
            if args['vf_load'] == "yes":
                RUN_DIR = RUN_DIR + '_' + 'vf'
                assert args['vf_type'] is not ""
                RUN_DIR = RUN_DIR + '_' + args['vf_type']
            if args['pol_load'] == "yes":
                RUN_DIR = RUN_DIR + '_' + 'pol'

            if args['vf_switch'] == "yes":
                RUN_DIR = RUN_DIR + '_' + 'switch'

            MODEL_DIR = os.path.join(RUN_DIR, 'model')
            FIGURE_DIR = os.path.join(RUN_DIR, 'figure')
            RESULT_DIR = os.path.join(RUN_DIR, 'result')
        else:
            raise ValueError("unknown algorithm!!")

        args['RUN_DIR'] = RUN_DIR
        args['MODEL_DIR'] = MODEL_DIR
        args['FIGURE_DIR'] = FIGURE_DIR
        args['RESULT_DIR'] = RESULT_DIR
        # ---------------------------

        # ------- logger initialize and configuration -------
        logger.configure(dir=args['RUN_DIR'])
        # ---------------------------------------------------

        # Initialize environment and reward type
        env = gym.make(args['gym_env'], reward_type=args['reward_type'], set_additional_goal=args['set_additional_goal'])


        # Set random seed in hope to reproductability
        env.seed(args['seed'])
        np.random.seed(args['seed'])
        tf.set_random_seed(args['seed'])


        logger.record_tabular("algo", args['algo'])
        logger.record_tabular("env", args['gym_env'])
        logger.record_tabular("env.set_additional_goal", env.set_additional_goal)
        logger.record_tabular("env.reward_type", env.reward_type)
        logger.dump_tabular()

        if args['algo'] == "ppo":
            # Make necessary directories
            maybe_mkdir(args['RUN_DIR'])
            maybe_mkdir(args['MODEL_DIR'])
            maybe_mkdir(args['FIGURE_DIR'])
            maybe_mkdir(args['RESULT_DIR'])
            ppo_params_json = os.environ['PROJ_HOME_3']+'/ppo1/ppo_params.json'

            # Start to train the policy from scratch
            trained_policy = run(env=env, algorithm=ppo, params=ppo_params_json, args=args)
            trained_policy.save_model(args['MODEL_DIR'])

            # Load model to collect more data in order to calculate gradients
            # LOAD_DIR = os.environ['PROJ_HOME_3'] + '/runs_log_tests/19-Jul-2020_02-18-09PlanarQuadEnv-v0_hand_craft_ppo/model'
            # print("load policy iteration: ", args['policy_iteration'])
            # trained_policy = run(env=env, algorithm=ppo, params=ppo_params_json, load=True, loadpath=LOAD_DIR, loaditer=args['policy_iteration'], args=args)
            # trained_policy.save_model(args['MODEL_DIR'])

            # Load model and continue training
            # LOAD_DIR = os.environ['PROJ_HOME_3'] + '/runs_log_tests/07-Jul-2020_00-41-11PlanarQuadEnv-v0_hand_craft_ppo_vf_boltzmann/model'
            # trained_policy = run(env=env, algorithm=ppo, params=ppo_params_json, load=True, loadpath=LOAD_DIR, loaditer=150, args=args)
            # trained_policy.save_model(args['MODEL_DIR'])

            # Load pre-trained model for evaluation
            # LOAD_DIR = os.environ['PROJ_HOME_3'] + '/runs_log_tests/grad_norm_0.5_kl_0.015_std_0.5_baseline/27-Jan-2020_01-44-06DubinsCarEnv-v0_hand_craft_ppo/model'
            # LOAD_DIR = os.environ['PROJ_HOME_3'] + '/runs_log_tests/grad_norm_0.5_kl_0.015_std_0.5_fixed_value_vi/23-Jan-2020_00-13-24DubinsCarEnv-v0_hand_craft_ppo_vf_boltzmann/model'
            # LOAD_DIR = os.environ[
            #                'PROJ_HOME_3'] + '/runs_log_tests/quad_task_exploration/quad_task_air_space_202002_Francis_goal_angle_0_60/baseline/29-Apr-2020_21-12-34PlanarQuadEnv-v0_hand_craft_ppo/model'
            # LOAD_DIR = os.environ[
            #                'PROJ_HOME_3'] + '/runs_log_tests/quad_task_exploration/quad_task_air_space_202002_Francis_goal_angle_0_60/vi_fixed/25-Apr-2020_01-53-24PlanarQuadEnv-v0_hand_craft_ppo_vf_boltzmann/model'
            # eval_policy = run(env=env, algorithm=ppo, params=ppo_params_json, load=True, loadpath=LOAD_DIR, loaditer=180, args=args)

        else:
            raise ValueError("arg algorithm is invalid!")


# this little code block is to verify the effectiveness of random seed set at top of this file.
# if __name__ == "__main__":
#     print("numpy generate a random number:", np.random.uniform(0, 1))
#     tf_rand = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
#
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         print("tensorflow generate a random number:", sess.run(tf_rand))
