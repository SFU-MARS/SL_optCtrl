import sys,os
sys.path.append(os.environ['PROJ_HOME_3'] + "/tests")
import numpy as np
import gym
import tests.test_vpg_off_policy.core as test_core
import tests.test_vpg_off_policy.vpg as test_vpg
from keras.models import load_model
from gym_foo import gym_foo
import pandas as pd
import pickle

from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PlanarQuadEnv-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--off_policy_update', type=str, default='true')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    print("Testing vpg using off-policy data ...")

    test_vpg.vpg(lambda: gym.make(args.env, reward_type='hand_craft', set_additional_goal='None'),
                 actor_critic=test_core.mlp_actor_critic if args.off_policy_update == 'false' else test_core.mlp_actor_critic_val_init,
                 ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
                 seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
                 logger_kwargs=logger_kwargs, off_policy_update=args.off_policy_update)






