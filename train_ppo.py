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

def train(env, algorithm, args, params=None, load=False, loadpath=None, loaditer=None, save_obs=False):

    if algorithm == ppo:
        assert args['gym_env'] == "DubinsCarEnv-v0" or args['gym_env'] == "PlanarQuadEnv-v0"

        # Initialize policy
        ppo.create_session()
        init_policy = ppo.create_policy('pi', env, vf_load=True if args['vf_load'] == "yes" else False, pol_load=True if args['pol_load'] == "yes" else False)
        ppo.initialize()

        if load and loadpath is not None and loaditer is not None:
            # load trained policy
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
            num_iters = d.get('num_iters')
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

        # record performance data
        overall_perf = list()
        ppo_reward = list()
        ppo_length = list()
        suc_percents = list()
        wall_clock_time = list()

        best_suc_percent = 0
        perf_flag = False

        eval_ppo_reward = list()
        eval_suc_percents = list()
        # index for num_iters loop
        i = 1
        while i <= num_iters:
            wall_clock_time.append(time())
            logger.info('overall training iteration %d' %i)
            # each learning step contains "num_ppo_iters" ppo-learning steps.
            # each ppo-learning steps == ppo-learning on single episode
            # each single episode is a single markov chain which contains many states, actions, rewards.
            pi, ep_mean_length, ep_mean_reward, suc_percent = algorithm.ppo_learn(env=env, policy=pi, timesteps_per_actorbatch=timesteps_per_actorbatch,
                                                                     clip_param=clip_param, entcoeff=entcoeff, optim_epochs=optim_epochs,
                                                                     optim_stepsize=optim_stepsize, optim_batchsize=optim_batchsize,
                                                                     gamma=gamma, lam=lam, max_iters=max_iters, schedule='constant', save_obs=save_obs)

            ppo_length.extend(ep_mean_length)
            ppo_reward.extend(ep_mean_reward)
            suc_percents.append(suc_percent)

            # perf_metric = evaluate()
            # overall_perf.append(perf_metric)
            # print('[Overall Iter %d]: perf_metric = %.2f' % (i, perf_metric))

            pi.save_model(args['MODEL_DIR'], iteration=i)
            plot_performance(range(len(ppo_reward)), ppo_reward, ylabel=r'avg reward per ppo-learning step',
                             xlabel='ppo iteration', figfile=os.path.join(args['FIGURE_DIR'], 'ppo_reward'), title='TRAIN')
            plot_performance(range(len(suc_percents)), suc_percents,
                             ylabel=r'overall success percentage per algorithm step',
                             xlabel='algorithm iteration', figfile=os.path.join(args['FIGURE_DIR'], 'success_percent'), title="TRAIN")

            # --- for plotting evaluation perf on success rate using early stopping trick ---
            logger.record_tabular('suc_percent', suc_percent)
            logger.record_tabular('best_suc_percent', best_suc_percent)
            logger.record_tabular('perf_flag', perf_flag)
            logger.dump_tabular()

            if suc_percent >= best_suc_percent:
                best_suc_percent = suc_percent
                pi.save_model(args['MODEL_DIR'], iteration='best')


            pi.load_model(args['MODEL_DIR'], iteration='best')
            _, _, eval_ep_mean_reward, eval_suc_percent, _, _ = algorithm.ppo_eval(env, pi, timesteps_per_actorbatch//2, max_iters=5, stochastic=False)

            eval_ppo_reward.extend(eval_ep_mean_reward)
            eval_suc_percents.append(eval_suc_percent)

            plot_performance(range(len(eval_ppo_reward)), eval_ppo_reward, ylabel=r'avg reward per ppo-eval step',
                             xlabel='ppo iteration', figfile=os.path.join(args['FIGURE_DIR'], 'eval_ppo_reward'), title='EVAL')
            plot_performance(range(len(eval_suc_percents)), eval_suc_percents,
                             ylabel=r'overall eval success percentage per algorithm step',
                             xlabel='algorithm iteration', figfile=os.path.join(args['FIGURE_DIR'], 'eval_success_percent'),
                             title="EVAL")
            # -------------------------------------------------------------------------------



            # save data which is accumulated UNTIL iter i
            with open(args['RESULT_DIR'] + '/ppo_length_'+'iter_'+str(i)+'.pickle','wb') as f1:
                pickle.dump(ppo_length, f1)
            with open(args['RESULT_DIR'] + '/ppo_reward_'+'iter_'+str(i)+'.pickle','wb') as f2:
                pickle.dump(ppo_reward, f2)
            with open(args['RESULT_DIR'] + '/success_percent_' + 'iter_' + str(i) + '.pickle', 'wb') as fs:
                pickle.dump(suc_percents, fs)
            with open(args['RESULT_DIR'] + '/wall_clock_time_' + 'iter_' + str(i) + '.pickle', 'wb') as ft:
                pickle.dump(wall_clock_time, ft)

            # save evaluation data accumulated until iter i
            with open(args['RESULT_DIR'] + '/eval_ppo_reward_' + 'iter_' +str(i) + '.pickle','wb') as f_er:
                pickle.dump(eval_ppo_reward, f_er)
            with open(args['RESULT_DIR'] + '/eval_success_percent_' + 'iter_' + str(i) + '.pickle', 'wb') as f_es:
                pickle.dump(eval_suc_percents, f_es)

            # Incrementing our algorithm's loop counter
            i += 1

        # overall, we need plot the time-to-reach for the best policy so far.

        env.close()

        return pi

    elif algorithm == deepq:
        pass
    else:
        raise ValueError("Please input an valid algorithm")


if __name__ == "__main__":
    with tf.device('/gpu:1'):
        # ----- path setting ------
        parser = argparse.ArgumentParser()
        parser.add_argument("--gym_env", help="which gym environment to use.", type=str, default='PlanarQuadEnv-v0')
        parser.add_argument("--reward_type", help="which type of reward to use.", type=str, default='hand_craft')
        parser.add_argument("--algo", help="which type of algorithm to use.", type=str, default='ppo')
        parser.add_argument("--set_additional_goal", type=str, default="angle")
        parser.add_argument("--vf_load", type=str, default="no")
        parser.add_argument("--pol_load", type=str, default="no")
        args = parser.parse_args()
        args = vars(args)

        RUN_DIR = MODEL_DIR = FIGURE_DIR = RESULT_DIR = None
        if args['algo'] == "ppo":
            RUN_DIR = os.path.join(os.getcwd(), 'runs_log',
                                   strftime(
                                       '%d-%b-%Y_%H-%M-%S') + args['gym_env'] + '_' + args['reward_type'] + '_' + args['algo'])
            if args['vf_load'] == "yes":
                RUN_DIR = RUN_DIR + '_' + 'vf'
            if args['pol_load'] == "yes":
                RUN_DIR = RUN_DIR + '_' + 'pol'

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

            # Start to train the policy
            trained_policy = train(env=env, algorithm=ppo, params=ppo_params_json, args=args)
            trained_policy.save_model(args['MODEL_DIR'])
            #
            # LOAD_DIR = os.environ['PROJ_HOME_3'] + '/runs_icra/04-Sep-2019_08-59-16PlanarQuadEnv-v0_hand_craft_ppo/model'
            # trained_policy = train(env=env, algorithm=ppo, params=ppo_params_json, load=True, loadpath=LOAD_DIR, loaditer=3, args=args)

        else:
            raise ValueError("arg algorithm is invalid!")















