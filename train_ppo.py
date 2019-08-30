
import gym
from gym_foo import gym_foo
from gym import wrappers
from time import *

import ppo
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
        init_policy = ppo.create_policy('pi', env, vf_load=True)
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
            if suc_percent > 0.6:
                perf_flag = True
            if not perf_flag or env.reward_type != 'ttr':
                # less timesteps_per_actorbatch to make eval faster.
                _, _, eval_ep_mean_reward, eval_suc_percent, _, _ = algorithm.ppo_eval(env, pi, timesteps_per_actorbatch//2, max_iters=5, stochastic=False)
            else:

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

        # plot_performance(range(len(overall_perf)), overall_perf, ylabel=r'overall performance per algorithm step',
        #                  xlabel='algorithm iteration',
        #                  figfile=os.path.join(FIGURE_DIR, 'overall_perf'))

        # overall, we need plot the time-to-reach for the best policy so far.

        env.close()

        return pi

    elif algorithm == deepq:
        pass
        # assert args.gym_env == "DubinsCarEnv_dqn-v0" or args.gym_env == "PlanarQuadEnv_dqn-v0"
        # # do something about dqn training
        # tmp_path = MODEL_DIR + '/ep'
        #
        # continue_execution = False
        # resume_epoch = '200'
        # resume_path = tmp_path + resume_epoch
        # weights_path = resume_path + '.h5'
        # params_json = resume_path + '.json'
        #
        # epochs = steps = updateTargetNetwork = explorationRate = minibatch_size = learnStart = learningRate= \
        # discountFactor = memorySize = network_inputs = network_outputs = network_structure = current_epoch = None
        #
        # if not continue_execution:
        #     # Each time we take a sample and update our weights it is called a mini-batch.
        #     # Each time we run through the entire dataset, it's called an epoch.
        #     epochs = 1000
        #     steps = 1000
        #     updateTargetNetwork = 10000
        #     explorationRate = 1
        #     minibatch_size = 128
        #     learnStart = 64
        #     learningRate = 0.00025
        #     discountFactor = 0.99
        #     memorySize = 1000000
        #     network_inputs = env.state_dim + env.num_lasers
        #     # network_outputs = 21
        #     network_outputs = 25
        #     network_structure = [300,300]
        #     current_epoch = 0
        #
        #     deepQ = deepq.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        #     deepQ.initNetworks(network_structure)
        # else:
        #     # Load weights and parameter info.
        #     with open(params_json) as outfile:
        #         d = json.load(outfile)
        #         epochs = d.get('epochs')
        #         steps = d.get('steps')
        #         updateTargetNetwork = d.get('updateTargetNetwork')
        #         explorationRate = d.get('explorationRate')
        #         minibatch_size = d.get('minibatch_size')
        #         learnStart = d.get('learnStart')
        #         learningRate = d.get('learningRate')
        #         discountFactor = d.get('discountFactor')
        #         memorySize = d.get('memorySize')
        #         network_inputs = d.get('network_inputs')
        #         network_outputs = d.get('network_outputs')
        #         network_structure = d.get('network_structure')
        #         current_epoch = d.get('current_epoch')
        #
        #     deepQ = deepq.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        #     deepQ.initNetworks(network_structure)
        #
        #     deepQ.loadWeights(weights_path)
        #
        # env._max_episode_steps = steps
        # last100Scores = [0] * 100
        # last100ScoresIndex = 0
        # last100Filled = False
        # stepCounter = 0
        # highest_reward = 0
        #
        # start_time = time()
        #
        # # start iterating from 'current epoch'.
        # for epoch in np.arange(current_epoch + 1, epochs + 1, 1):
        #     observation = env.reset()
        #     cumulated_reward = 0
        #     done = False
        #     episode_step = 0
        #
        #     # run until env returns done
        #     while not done:
        #         # env.render()
        #         qValues = deepQ.getQValues(observation)
        #
        #         action = deepQ.selectAction(qValues, explorationRate)
        #
        #         newObservation, reward, done, suc, info = env.step(action)
        #
        #         cumulated_reward += reward
        #         if highest_reward < cumulated_reward:
        #             highest_reward = cumulated_reward
        #
        #         deepQ.addMemory(observation, action, reward, newObservation, done)
        #
        #         if stepCounter >= learnStart:
        #             if stepCounter <= updateTargetNetwork:
        #                 deepQ.learnOnMiniBatch(minibatch_size, False)
        #             else:
        #                 deepQ.learnOnMiniBatch(minibatch_size, True)
        #
        #         observation = newObservation
        #
        #         if done:
        #             last100Scores[last100ScoresIndex] = episode_step
        #             last100ScoresIndex += 1
        #             if last100ScoresIndex >= 100:
        #                 last100Filled = True
        #                 last100ScoresIndex = 0
        #             if not last100Filled:
        #                 print("EP " + str(epoch) + " - " + format(episode_step + 1) + "/" + str(
        #                     steps) + " Episode steps   Exploration=" + str(round(explorationRate, 2)))
        #             else:
        #                 m, s = divmod(int(time() - start_time), 60)
        #                 h, m = divmod(m, 60)
        #                 print("EP " + str(epoch) + " - " + format(episode_step + 1) + "/" + str(
        #                     steps) + " Episode steps - last100 Steps : " + str(
        #                     (sum(last100Scores) / len(last100Scores))) + " - Cumulated R: " + str(
        #                     cumulated_reward) + "   Eps=" + str(
        #                     round(explorationRate, 2)) + "     Time: %d:%02d:%02d" % (h, m, s))
        #                 if epoch % 100 == 0:
        #                     # save model weights and monitoring data every 100 epochs.
        #                     deepQ.saveModel(tmp_path + str(epoch) + '.h5')
        #                     # save simulation parameters.
        #                     # convert from numpy int64 type to python int type for json serialization
        #                     epoch = int(epoch)
        #                     parameter_keys = ['epochs', 'steps', 'updateTargetNetwork', 'explorationRate',
        #                                       'minibatch_size', 'learnStart', 'learningRate', 'discountFactor',
        #                                       'memorySize', 'network_inputs', 'network_outputs', 'network_structure',
        #                                       'current_epoch']
        #                     parameter_values = [epochs, steps, updateTargetNetwork, explorationRate, minibatch_size,
        #                                         learnStart, learningRate, discountFactor, memorySize, network_inputs,
        #                                         network_outputs, network_structure, epoch]
        #                     parameter_dictionary = dict(zip(parameter_keys, parameter_values))
        #                     with open(tmp_path + str(epoch) + '.json', 'w') as outfile:
        #                         json.dump(parameter_dictionary, outfile)
        #
        #         stepCounter += 1
        #         if stepCounter % updateTargetNetwork == 0:
        #             deepQ.updateTargetNetwork()
        #             print("updating target network")
        #
        #         episode_step += 1
        #
        #     explorationRate *= 0.995  # epsilon decay
        #     # explorationRate -= (2.0/epochs)
        #     explorationRate = max(0.05, explorationRate)
        #
        # env.close()
        # return 1

    else:
        raise ValueError("Please input an valid algorithm")


if __name__ == "__main__":
    with tf.device('/gpu:0'):
        # ----- path setting ------
        parser = argparse.ArgumentParser()
        parser.add_argument("--gym_env", help="which gym environment to use.", type=str, default='PlanarQuadEnv-v0')
        parser.add_argument("--reward_type", help="which type of reward to use.", type=str, default='hand_craft')
        parser.add_argument("--algo", help="which type of algorithm to use.", type=str, default='ppo')
        parser.add_argument("--set_additional_goal", type=str, default="angle")
        args = parser.parse_args()
        args = vars(args)

        RUN_DIR = MODEL_DIR = FIGURE_DIR = RESULT_DIR = None
        if args['algo'] == "ppo":
            RUN_DIR = os.path.join(os.getcwd(), 'runs_icra',
                                   args['gym_env'] + '_' + args['reward_type'] + '_' + args['algo'] + '_' + strftime(
                                       '%d-%b-%Y_%H-%M-%S'))
            MODEL_DIR = os.path.join(RUN_DIR, 'model')
            FIGURE_DIR = os.path.join(RUN_DIR, 'figure')
            RESULT_DIR = os.path.join(RUN_DIR, 'result')
        elif args['algo'] == "dqn":
            RUN_DIR = os.path.join(os.getcwd(), 'runs_icra',
                                   args['gym_env'] + '_' + args['reward_type'] + '_' + args['algo'] + '_' + strftime('%d-%b-%Y_%H-%M-%S'))
            MODEL_DIR = os.path.join(RUN_DIR, 'model')

        args['RUN_DIR'] = RUN_DIR
        args['MODEL_DIR'] = MODEL_DIR
        args['FIGURE_DIR'] = FIGURE_DIR
        args['RESULT_DIR'] = RESULT_DIR
        # ---------------------------

        # ------- logger initialize and configuration -------
        logger.configure(dir=args['RUN_DIR'])
        # ---------------------------------------------------

        # Initialize environment and reward type
        env = gym.make(args['gym_env'])
        env.reward_type = args['reward_type']
        env.set_additional_goal = args['set_additional_goal']
        logger.record_tabular("algo", args['algo'])
        logger.record_tabular("env", args['gym_env'])
        logger.record_tabular("env.set_additional_goal", env.set_additional_goal)
        logger.record_tabular("env.reward_type", env.reward_type)

        logger.dump_tabular()

        # Initialize brs engine. You also have to call reset_variables() after instance initialization
        if args['reward_type'] == 'ttr':

            if args['gym_env'] == 'DubinsCarEnv-v0' or args['gym_env'] == 'DubinsCarEnv_dqn-v0':
                brsEngine = DubinsCar_brs_engine()
                brsEngine.reset_variables()

            elif args['gym_env'] == 'PlanarQuadEnv-v0' or args['gym_env'] == 'PlanarQuadEnv_dqn-v0':
                brsEngine = Quadrotor_brs_engine()
                brsEngine.reset_variables()

            else:
                raise ValueError("invalid environment name for ttr reward!")

            # You have to assign the engine
            env.brsEngine = brsEngine

        elif args['reward_type'] in ['hand_craft','distance','distance_lambda_10','distance_lambda_1','distance_lambda_0.1']:
            pass
        else:
            raise ValueError("wrong type of reward")

        if args['algo'] == "ppo":
            # Make necessary directories
            maybe_mkdir(args['RUN_DIR'])
            maybe_mkdir(args['MODEL_DIR'])
            maybe_mkdir(args['FIGURE_DIR'])
            maybe_mkdir(args['RESULT_DIR'])
            ppo_params_json = os.environ['PROJ_HOME']+'/ppo_params.json'

            # Start to train the policy
            trained_policy = train(env=env, algorithm=ppo, params=ppo_params_json, args=args)
            trained_policy.save_model(args['MODEL_DIR'])
            #
            # LOAD_DIR = os.environ['PROJ_HOME'] + '/runs_icra/PlanarQuadEnv-v0_hand_craft_ppo_16-Aug-2019_11-37-22/model'
            # trained_policy = train(env=env, algorithm=ppo, params=ppo_params_json, load=True, loadpath=LOAD_DIR, loaditer=18)

        elif args['algo'] == "dqn":
            # Make necessary directories
            maybe_mkdir(args['RUN_DIR'])
            maybe_mkdir(args['MODEL_DIR'])
            flag = train(env=env, algorithm=deepq)


        else:
            raise ValueError("arg algorithm is invalid!")















