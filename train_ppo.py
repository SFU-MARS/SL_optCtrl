import sys
#sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
#sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')
import gym
from gym_foo import gym_foo
from gym import wrappers
from time import *

from ppo1 import ppo
import utils.liveplot as liveplot
from utils.plotting_performance import *
from baselines import logger

import argparse
from utils.utils import *
import json
import pickle

import gazebo_env
import copy

import tensorflow as tf



"""
def train(env, algorithm, args, params=None, load=False, loadpath=None, loaditer=None, save_obs=False):

    assert algorithm == ppo
    assert args['gym_env'] in ["AckermannEnv-v0", "PlanarQuadEnv-v0", "DubinsCarEnv-v0"]

    # Initialize policy
    ppo.create_session()
    init_policy = ppo.create_policy('pi', env, vf_load=True if args['vf_load'] == "yes" else False, pol_load=True if args['pol_load'] == "yes" else False)
    ppo.initialize()

    # load trained policy
    if load and loadpath is not None and loaditer is not None:
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
        num_ppo_iters = d.get('num_ppo_iters')
        timesteps_per_actorbatch = d.get('timesteps_per_actorbatch')
        clip_param = d.get('clip_param')
        entcoeff = d.get('entcoeff')
        optim_epochs = d.get('optim_epochs')
        optim_stepsize = d.get('optim_stepsize')
        optim_batchsize = d.get('optim_batchsize')
        gamma = d.get('gamma')
        lam = d.get('lam')
        max_iters = num_ppo_iters

    pi = algorithm.ppo_learn(env=env, policy=pi, timesteps_per_actorbatch=timesteps_per_actorbatch,
                             clip_param=clip_param, entcoeff=entcoeff, optim_epochs=optim_epochs,
                             optim_stepsize=optim_stepsize, optim_batchsize=optim_batchsize,
                             gamma=gamma, lam=lam,  args=args, max_iters=max_iters, schedule='constant', save_obs=save_obs)

    env.close()

    # # record performance data
    # train_reward = list() # a list with size "max_iters" to save avg_reward at each iter
    # train_length = list() # a list with size "max_iters" to save avg_length at each iter
    # eval_success_rate = list() # a list with size
    # wall_clock_time = list()
    #
    # best_suc_percent = 0
    # # perf_flag = False
    #
    # eval_ppo_reward = list()
    # eval_suc_percents = list()

    # pi, ep_mean_length, ep_mean_reward, suc_percent = algorithm.ppo_learn(env=env, policy=pi,timesteps_per_actorbatch=timesteps_per_actorbatch,
    #                                                              clip_param=clip_param, entcoeff=entcoeff, optim_epochs=optim_epochs,
    #                                                              optim_stepsize=optim_stepsize, optim_batchsize=optim_batchsize,
    #                                                              gamma=gamma, lam=lam,  args=args, max_iters=max_iters, schedule='constant', save_obs=save_obs)
    #
    # # ppo_length.extend(ep_mean_length)
    # ppo_reward.extend(ep_mean_reward)
    # suc_percents.extend(suc_percent)

    # save trained model at current iter
    # pi.save_model(args['MODEL_DIR'], iteration=i)
    # plot required training data accumulated until current iter
    # plot_performance(range(len(ppo_reward)), ppo_reward, ylabel=r'avg reward per ppo-learning step',
    #                  xlabel='ppo iteration', figfile=os.path.join(args['FIGURE_DIR'], 'ppo_reward'), title='TRAIN')
    # plot_performance(range(len(suc_percents)), suc_percents,
    #                  ylabel=r'overall success percentage per algorithm step',
    #                  xlabel='algorithm iteration', figfile=os.path.join(args['FIGURE_DIR'], 'success_percent'), title="TRAIN")






    # pre-evaluation before everything starts
    # logger.info('pre-evaluation before everything starts')
    # pi.load_model(args['MODEL_DIR'], iteration=0)
    # _, _, eval_ep_mean_reward, eval_suc_percent, _, _ = algorithm.ppo_eval(env, pi,
    #                                                                        timesteps_per_actorbatch // 2,
    #                                                                        max_iters=5, stochastic=False)
    # eval_ppo_reward.extend(eval_ep_mean_reward)
    # eval_suc_percents.append(eval_suc_percent)

    # -----------------------------------------------------------
    # # index for num_iters loop
    # i = 1
    # while i <= num_iters:
    #     wall_clock_time.append(time())
    #     logger.info('overall training iteration %d' %i)
    #     # each learning step contains "num_ppo_iters" ppo-learning steps.
    #     # each ppo-learning steps == ppo-learning on single episode
    #     # each single episode is a single markov chain which contains many states, actions, rewards.
    #     # Now suc_percent is a list containing 'max_iters' success_rate
    #     pi, ep_mean_length, ep_mean_reward, suc_percent = algorithm.ppo_learn(env=env, policy=pi, timesteps_per_actorbatch=timesteps_per_actorbatch,
    #                                                              clip_param=clip_param, entcoeff=entcoeff, optim_epochs=optim_epochs,
    #                                                              optim_stepsize=optim_stepsize, optim_batchsize=optim_batchsize,
    #                                                              gamma=gamma, lam=lam,  args=args, max_iters=max_iters, schedule='constant', save_obs=save_obs)
    #
    #     ppo_length.extend(ep_mean_length)
    #     ppo_reward.extend(ep_mean_reward)
    #     suc_percents.extend(suc_percent)
    #
    #     # save trained model at current iter
    #     pi.save_model(args['MODEL_DIR'], iteration=i)
    #     # plot required training data accumulated until current iter
    #     plot_performance(range(len(ppo_reward)), ppo_reward, ylabel=r'avg reward per ppo-learning step',
    #                      xlabel='ppo iteration', figfile=os.path.join(args['FIGURE_DIR'], 'ppo_reward'), title='TRAIN')
    #     plot_performance(range(len(suc_percents)), suc_percents,
    #                      ylabel=r'overall success percentage per algorithm step',
    #                      xlabel='algorithm iteration', figfile=os.path.join(args['FIGURE_DIR'], 'success_percent'), title="TRAIN")
    #
    #     # save training data which is accumulated UNTIL iter i
    #     with open(args['RESULT_DIR'] + '/ppo_length_' + 'iter_' + str(i) + '.pickle', 'wb') as f1:
    #         pickle.dump(ppo_length, f1)
    #     with open(args['RESULT_DIR'] + '/ppo_reward_' + 'iter_' + str(i) + '.pickle', 'wb') as f2:
    #         pickle.dump(ppo_reward, f2)
    #     with open(args['RESULT_DIR'] + '/success_percent_' + 'iter_' + str(i) + '.pickle', 'wb') as fs:
    #         pickle.dump(suc_percents, fs)
    #     with open(args['RESULT_DIR'] + '/wall_clock_time_' + 'iter_' + str(i) + '.pickle', 'wb') as ft:
    #         pickle.dump(wall_clock_time, ft)
    #
    #
    #
    #     # # --- for plotting evaluation perf on success rate using early stopping trick ---
    #     # logger.record_tabular('suc_percent', suc_percent)
    #     # logger.record_tabular('best_suc_percent', best_suc_percent)
    #     # # logger.record_tabular('perf_flag', perf_flag)
    #     # logger.dump_tabular()
    #     #
    #     # # here we need compare 'eval_suc_percent' with 'best_suc_percent'
    #     # # not previous suc_percent
    #     # if eval_suc_percent >= best_suc_percent:
    #     #     best_suc_percent = suc_percent
    #     #     pi.save_model(args['MODEL_DIR'], iteration='best')
    #     #
    #     #
    #     # pi.load_model(args['MODEL_DIR'], iteration='best')
    #     # _, _, eval_ep_mean_reward, eval_suc_percent, _, _ = algorithm.ppo_eval(env, pi, timesteps_per_actorbatch//2, max_iters=5, stochastic=False)
    #     #
    #     # eval_ppo_reward.extend(eval_ep_mean_reward)
    #     # eval_suc_percents.append(eval_suc_percent)
    #     #
    #     # plot_performance(range(len(eval_ppo_reward)), eval_ppo_reward, ylabel=r'avg reward per ppo-eval step',
    #     #                  xlabel='ppo iteration', figfile=os.path.join(args['FIGURE_DIR'], 'eval_ppo_reward'), title='EVAL')
    #     # plot_performance(range(len(eval_suc_percents)), eval_suc_percents,
    #     #                  ylabel=r'overall eval success percentage per algorithm step',
    #     #                  xlabel='algorithm iteration', figfile=os.path.join(args['FIGURE_DIR'], 'eval_success_percent'),
    #     #                  title="EVAL")
    #     # # -------------------------------------------------------------------------------
    #
    #
    #
    #
    #     # # save evaluation data accumulated until iter i
    #     # with open(args['RESULT_DIR'] + '/eval_ppo_reward_' + 'iter_' +str(i) + '.pickle','wb') as f_er:
    #     #     pickle.dump(eval_ppo_reward, f_er)
    #     # with open(args['RESULT_DIR'] + '/eval_success_percent_' + 'iter_' + str(i) + '.pickle', 'wb') as f_es:
    #     #     pickle.dump(eval_suc_percents, f_es)
    #
    #     # Incrementing our algorithm's loop counter
    #     i += 1
    #
    # # overall, we need plot the time-to-reach for the best policy so far.
    #
    # env.close()
    # --------------------------------------------------------------------------

    return pi

"""



def run(env, algorithm, args, params=None, load=False, loadpath=None, loaditer=None, save_obs=False):

    assert algorithm == ppo
    assert args['gym_env'] in ["AckermannEnv-v0", "PlanarQuadEnv-v0", "DubinsCarEnv-v0"]

    # Initialize policy
    ppo.create_session()
    init_policy = ppo.create_policy('pi', env, args=args, vf_load=True if args['vf_load'] == "yes" else False, pol_load=True if args['pol_load'] == "yes" else False)
    ppo.initialize()

    # load trained policy
    if load and loadpath is not None and loaditer is not None:
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
        num_ppo_iters = d.get('num_ppo_iters')
        timesteps_per_actorbatch = d.get('timesteps_per_actorbatch')
        clip_param = d.get('clip_param')
        entcoeff = d.get('entcoeff')
        optim_epochs = d.get('optim_epochs')
        optim_stepsize = d.get('optim_stepsize')
        optim_batchsize = d.get('optim_batchsize')
        gamma = d.get('gamma')
        lam = d.get('lam')
        max_iters = num_ppo_iters

    if not load:
        pi = algorithm.ppo_learn(env=env, policy=pi, timesteps_per_actorbatch=timesteps_per_actorbatch,
                                 clip_param=clip_param, entcoeff=entcoeff, optim_epochs=optim_epochs,
                                 optim_stepsize=optim_stepsize, optim_batchsize=optim_batchsize,
                                 gamma=gamma, lam=lam,  args=args, max_iters=max_iters, schedule='constant', save_obs=save_obs)
    else:
        pi = algorithm.ppo_eval(env=env, policy=pi, timesteps_per_actorbatch=timesteps_per_actorbatch,max_iters=5)
    env.close()
    return pi

if __name__ == "__main__":
    with tf.device('/gpu:1'):
        # ----- path setting ------
        parser = argparse.ArgumentParser()
        parser.add_argument("--gym_env", help="which gym environment to use.", type=str, default='DubinsCarEnv-v0')
        parser.add_argument("--reward_type", help="which type of reward to use.", type=str, default='hand_craft')
        parser.add_argument("--algo", help="which type of algorithm to use.", type=str, default='ppo')
        parser.add_argument("--set_additional_goal", type=str, default="angle")
        parser.add_argument("--vf_load", type=str, default="yes")
        parser.add_argument("--pol_load", type=str, default="no")
        parser.add_argument("--vf_type", type=str, default="boltzmann")
        parser.add_argument("--vf_switch", type=str, default="no")
        args = parser.parse_args()
        args = vars(args)

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

            # Load pre-trained model for evaluation
            # LOAD_DIR = os.environ['PROJ_HOME_3'] + '/runs_log_tests/grad_norm_0.5_kl_0.015_std_0.5_baseline/27-Jan-2020_01-44-06DubinsCarEnv-v0_hand_craft_ppo/model'
            # LOAD_DIR =  os.environ['PROJ_HOME_3'] + '/runs_log_tests/grad_norm_0.5_kl_0.015_std_0.5_fixed_value_vi/23-Jan-2020_00-13-24DubinsCarEnv-v0_hand_craft_ppo_vf_boltzmann/model'


            # eval_policy = run(env=env, algorithm=ppo, params=ppo_params_json, load=True, loadpath=LOAD_DIR, loaditer=180, args=args)

        else:
            raise ValueError("arg algorithm is invalid!")















