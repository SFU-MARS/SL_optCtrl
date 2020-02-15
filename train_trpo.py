import sys
import gym
from gym_foo import gym_foo
from gym import wrappers
from time import *

from utils.plotting_performance import *
from baselines import logger

import argparse
from utils.utils import *
import json
import pickle

import gazebo_env
import copy

import tensorflow as tf

if __name__ == "__main__":
    # ----- path setting ------
    parser = argparse.ArgumentParser()
    parser.add_argument("--gym_env", help="which gym environment to use.", type=str, default='DubinsCarEnv-v0')
    parser.add_argument("--reward_type", help="which type of reward to use.", type=str, default='hand_craft')
    parser.add_argument("--set_additional_goal", type=str, default="angle")
    parser.add_argument("--vf_load", type=str, default="yes")
    parser.add_argument("--pol_load", type=str, default="no")
    parser.add_argument("--vf_type", type=str, default="boltzmann")
    parser.add_argument("--vf_switch", type=str, default="no")
    args = parser.parse_args()
    args = vars(args)


    RUN_DIR = os.path.join(os.getcwd(), 'runs_log_tests',
                           strftime('%d-%b-%Y_%H-%M-%S') + args['gym_env'] + '_' + args['reward_type'] + '_' + 'trpo')
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

    logger.record_tabular("algo", 'trpo')
    logger.record_tabular("env", args['gym_env'])
    logger.record_tabular("env.set_additional_goal", env.set_additional_goal)
    logger.record_tabular("env.reward_type", env.reward_type)
    logger.dump_tabular()


    # Make necessary directories
    maybe_mkdir(args['RUN_DIR'])
    maybe_mkdir(args['MODEL_DIR'])
    maybe_mkdir(args['FIGURE_DIR'])
    maybe_mkdir(args['RESULT_DIR'])

    import trpo.trpo as trpo
    import trpo.core as core
    from spinup.utils.mpi_tools import mpi_fork
    mpi_fork(1)  # run parallel code with mpi

    print(args['pol_load'])
    assert args['pol_load'] == 'no'
    if args['vf_load'] == 'no':
        # Start to train the policy from scratch
        logger.log("we are running with no value initialization at all")
        trpo.trpo(lambda: env, actor_critic=core.mlp_actor_critic,
             ac_kwargs=dict(hidden_sizes=[64] * 2), gamma=0.99,
             seed=0, steps_per_epoch=1024, epochs=300, main_kwargs=args)
    elif args['vf_load'] == 'yes':
        assert args['vf_type'] in ['boltzmann', 'mpc']
        # Start to train the policy from scratch
        logger.log("we are running with {} value initializaiton".format(args['vf_type']))
        trpo.trpo(lambda: env, actor_critic=core.mlp_actor_critic_vinit,
                  ac_kwargs=dict(hidden_sizes=[64] * 2, vinit=args['vf_type'], env=args['gym_env']),
                  gamma=0.99, seed=0, steps_per_epoch=1024, epochs=300, main_kwargs=args)
    else:
        raise ValueError("invalid arguments")