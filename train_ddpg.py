#!/local-scratch/xlv/miniconda3/envs/py35_no_specific/bin/python3.5


import argparse
import time
import os
import logging
from baselines import logger
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
import ddpg.training as training
from ddpg.models import Actor, Critic
from ddpg.memory import Memory
from ddpg.noise import *
from ddpg import bench


import gym
from gym_foo import gym_foo

import tensorflow as tf
from mpi4py import MPI
from utils.utils import *
import time

# from brs_engine.DubinsCar_brs_engine import *
# from brs_engine.PlanarQuad_brs_engine import *
from utils.plotting_performance import *


def run(env_id, seed, noise_type, layer_norm, evaluation, **kwargs):
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    # Create envs.
    env = gym.make(env_id)

    # ---------- AMEND: specific setting for brsEngine -----------
    print("kwargs", kwargs)
    env.reward_type = kwargs['reward_type']
    env.set_additional_goal = kwargs['set_additional_goal']
    kwargs.pop('reward_type', None)
    kwargs.pop('set_additional_goal', None)
    brsEngine = None
    if env.reward_type == 'ttr':
        if env_id == 'DubinsCarEnv-v0':
            brsEngine = DubinsCar_brs_engine()
            brsEngine.reset_variables()
        elif env_id == 'PlanarQuadEnv-v0':
            brsEngine = Quadrotor_brs_engine()
            brsEngine.reset_variables()
        else:
            raise ValueError("invalid environment name for ttr reward!")
        # You have to assign the engine!
        env.brsEngine = brsEngine
    # -----------------------------------------------------------

    # env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))

    if evaluation and rank==0:
        eval_env = gym.make(env_id)
        # ---------- AMEND: specific setting for brsEngine -----------
        eval_env.brsEngine = brsEngine
        # ------------------------------------------------------------

        eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'))
        env = bench.Monitor(env, None)
    else:
        eval_env = None

    # Parse noise_type
    action_noise = None
    param_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.
    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    critic = Critic(layer_norm=layer_norm)
    actor = Actor(nb_actions, layer_norm=layer_norm)

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()

    training.train(env=env, eval_env=eval_env, param_noise=param_noise, action_noise=action_noise, actor=actor, critic=critic, memory=memory, **kwargs)
    env.close()

    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env_id', type=str, default='PlanarQuadEnv-v0')
    boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-epochs', type=int, default=600)  # with default settings, perform 1M steps total
    # AMEND: change cycles to 10 to make it faster. original it's 20
    # AMEND: for simple car, adjust it to 5, then increase nb_rollout_steps to 200
    parser.add_argument('--nb-epoch-cycles', type=int, default=10)
    parser.add_argument('--nb-train-steps', type=int, default=50)  # per epoch cycle and MPI worker
    # actually, nb-eval_steps is not used anywhere
    parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-rollout-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str, default='adaptive-param_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--num-timesteps', type=int, default=None)
    boolean_flag(parser, 'evaluation', default=False)


    # AMEND: special argument added by xlv
    parser.add_argument('--reward_type', type=str, default='ttr')
    parser.add_argument('--set_additional_goal', type=str, default='angle')
    parser.add_argument('--algo', type=str, default='ddpg')
    boolean_flag(parser, 'restore', default=False)
    parser.add_argument('--restore_path', type=str, default=None)
    args = parser.parse_args()


    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if args.num_timesteps is not None:
        assert(args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles * args.nb_rollout_steps)
    dict_args = vars(args)
    del dict_args['num_timesteps']
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    # print(args)
    if args['env_id'] == 'DubinsCarEnv-v0':
        args['nb_epoch_cycles'] = 5
        args['nb_rollout_steps'] = 200
    print(args)
    # ---------------------- PATH SETTING ----------------------------
    RUN_DIR = MODEL_DIR = FIGURE_DIR = RESULT_DIR = None
    RUN_DIR = os.path.join(os.getcwd(), 'runs_log',
                           time.strftime(
                               '%d-%b-%Y_%H-%M-%S') + '_' + args['env_id'] + '_' + args['reward_type'] + '_' + args['algo'])
    MODEL_DIR = os.path.join(RUN_DIR, 'model')
    FIGURE_DIR = os.path.join(RUN_DIR, 'figure')
    RESULT_DIR = os.path.join(RUN_DIR, 'result')
    # ----------------------------------------------------------------
    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure(dir=RUN_DIR)

    # make necessary directories
    maybe_mkdir(RUN_DIR)
    maybe_mkdir(MODEL_DIR)
    maybe_mkdir(FIGURE_DIR)
    maybe_mkdir(RESULT_DIR)

    args['RUN_DIR'] = RUN_DIR
    args['MODEL_DIR'] = MODEL_DIR
    args['FIGURE_DIR'] = FIGURE_DIR
    args['RESULT_DIR'] = RESULT_DIR

    # Run actual script.
    print("args:", args)
    run(**args)

    


