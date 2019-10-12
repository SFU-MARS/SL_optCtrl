import os
import time
from collections import deque
import pickle

# from baselines.ddpg.ddpg import DDPG
from ddpg.ddpg import DDPG
import baselines.common.tf_util as U


from baselines import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI

from utils.plotting_performance import *
import pandas as pd

def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps,batch_size, memory,
    tau=0.01, eval_env=None, param_noise_adaption_interval=50, **kwargs):

    # print("kwargs:",kwargs)

    rank = MPI.COMM_WORLD.Get_rank()
    print("rank:",rank)
    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))
    # agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
    #     gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
    #     batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
    #     actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
    #     reward_scale=reward_scale)
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=128, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    if rank == 0:
        saver = tf.train.Saver()
    else:
        saver = None

    step = 0
    episode = 0

    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)

    with U.single_threaded_session() as sess:
        # Prepare everything.
        # --------------- AMEND: For saving and restoring the model. added by xlv ------------------
        if kwargs['restore'] == True and kwargs['restore_path'] != None:
            logger.info("Restoring from saved model")
            saver = tf.train.import_meta_graph(restore_path + "trained_model.meta")
            saver.restore(sess, tf.train.latest_checkpoint(restore_path))
        else:
            logger.info("Starting from scratch!")
            sess.run(tf.global_variables_initializer())
        # ----------------------------------------------------------------------------------------
        agent.initialize(sess)
        sess.graph.finalize()

        # --------------- AMEND: for warm-starting from demonstration data, added by xlv ------------
        # # 1. store all MPC data into replay buffer
        logger.info("Warm-starting for DDPG before real learning!")
        dataset_path = os.environ['PROJ_HOME_3'] + "/data/quad/polFunc_filled.csv"
        column_names = ['x', 'vx', 'z', 'vz', 'phi', 'w', 'a1', 'a2', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
        raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=",",
                                  skipinitialspace=True, skiprows=1)
        dataset = raw_dataset.copy()
        labelset = pd.concat([dataset.pop(x) for x in ['a1', 'a2']], 1)
        dataset_stats = dataset.describe().transpose()
        # normalize obs
        normed_dataset = (dataset - dataset_stats['mean']) / dataset_stats['std']
        normed_dataset = np.clip(normed_dataset, -5, 5)
        nrows = len(dataset)
        idx = 0
        counter = 0
        while idx < nrows:
            obs = dataset.loc[idx, ['x', 'vx', 'z', 'vz', 'phi', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']]
            normed_obs = normed_dataset.loc[idx, ['x', 'vx', 'z', 'vz', 'phi', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']]
            action = labelset.loc[idx, ['a1', 'a2']]
            # re-scale action, from [0,12] -> [-1,1]
            action = [-1 + (1 - (-1)) * (a_i - 0) / (12 - 0) for a_i in action]
            r = 1000 if env._in_goal(obs[0:6]) else 0
            new_normed_obs = normed_dataset.loc[
                idx + 1, ['x', 'vx', 'z', 'vz', 'phi', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']]
            done = True if r == 1000 else False
            agent.store_transition(normed_obs, action, r, new_normed_obs, done)
            counter = counter + 1
            idx = idx + 1
            if counter == 170:
                counter = 0
                idx = idx + 30



            print("proceeding {}th row and {}th row".format(idx, idx+1))
            print("normed obs:", normed_obs)
            print("obs:",obs)
            print("action:", action)

        logger.info("replay buffer size is {}".format(agent.memory.nb_entries))
        logger.info("loading MPC data into replay buffer successful!!")
        # 2. Do pre-training only using the MPC data in replay buffer
        for t_train in range(1000):
            # Adapt param noise, if necessary.
            if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                distance = agent.adapt_param_noise()
                # epoch_adaptive_distances.append(distance)
            cl, al = agent.train()
            logger.info("critic loss at current training step is {}".format(cl))
            logger.info("actor loss at current training step is {}".format(al))
            agent.update_target_net()
            logger.info("training step {}".format(t_train))
        # -------------------------------------------------------------------------------------------

        agent.reset()
        obs = eval_obs = env.reset()

        # if eval_env is not None:
        #     eval_obs = eval_env.reset()
        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        t = 0

        epoch = 0
        start_time = time.time()

        epoch_episode_rewards = []
        epoch_episode_steps = []


        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0

        # every 30 epochs plot statistics and save it.
        nb_epochs_unit = 30
        ddpg_rewards = []
        eval_ddpg_rewards = []

        ddpg_suc_percents = []
        eval_suc_percents = []

        # ---- AMEND: added by xlv to calculate success percent -----
        suc_num = 0
        episode_num = 0
        # -----------------------------------------------------------
        for epoch in range(nb_epochs):
            for cycle in range(nb_epoch_cycles):
                # Perform rollouts.
                for t_rollout in range(nb_rollout_steps):
                    # Predict next action.
                    action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                    assert action.shape == env.action_space.shape

                    # Execute next action.
                    if rank == 0 and render:
                        env.render()
                    assert max_action.shape == action.shape
                    # new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    new_obs, r, done, suc, info = env.step(max_action * action)
                    t += 1
                    if rank == 0 and render:
                        env.render()
                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    agent.store_transition(obs, action, r, new_obs, done)
                    obs = new_obs

                    if done:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        episode_reward = 0.
                        episode_step = 0
                        epoch_episodes += 1
                        episodes += 1
                        # --- AMEND: added by xlv to calculate success percent ---
                        episode_num += 1
                        if suc:
                            suc_num += 1
                        # -------------------------------------------------------
                        agent.reset()
                        obs = env.reset()

                # Train.
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                for t_train in range(nb_train_steps):
                    # Adapt param noise, if necessary.
                    if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise()
                        epoch_adaptive_distances.append(distance)

                    cl, al = agent.train()
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()

                # Evaluate.
                # eval_episode_rewards = []
                # eval_qs = []
                # if eval_env is not None:
                #     eval_episode_reward = 0.
                #     for t_rollout in range(nb_eval_steps):
                #         eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                #         eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                #         if render_eval:
                #             eval_env.render()
                #         eval_episode_reward += eval_r
                #
                #         eval_qs.append(eval_q)
                #         if eval_done:
                #             eval_obs = eval_env.reset()
                #             eval_episode_rewards.append(eval_episode_reward)
                #             eval_episode_rewards_history.append(eval_episode_reward)
                #             eval_episode_reward = 0.

            mpi_size = MPI.COMM_WORLD.Get_size()
            # Log stats.
            # XXX shouldn't call np.mean on variable length lists
            duration = time.time() - start_time
            stats = agent.get_stats()
            combined_stats = stats.copy()
            combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
            combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
            combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
            combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
            combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
            combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
            combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
            combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
            combined_stats['total/duration'] = duration
            combined_stats['total/steps_per_second'] = float(t) / float(duration)
            combined_stats['total/episodes'] = episodes
            combined_stats['rollout/episodes'] = epoch_episodes
            combined_stats['rollout/actions_std'] = np.std(epoch_actions)
            # Evaluation statistics.
            # if eval_env is not None:
            #     combined_stats['eval/return'] = eval_episode_rewards
            #     combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
            #     combined_stats['eval/Q'] = eval_qs
            #     combined_stats['eval/episodes'] = len(eval_episode_rewards)
            def as_scalar(x):
                if isinstance(x, np.ndarray):
                    assert x.size == 1
                    return x[0]
                elif np.isscalar(x):
                    return x
                else:
                    raise ValueError('expected scalar, got %s'%x)
            combined_stats_sums = MPI.COMM_WORLD.allreduce(np.array([as_scalar(x) for x in combined_stats.values()]))
            combined_stats = {k : v / mpi_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

            # Total statistics.
            combined_stats['total/epochs'] = epoch + 1
            combined_stats['total/steps'] = t

            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])
            logger.dump_tabular()
            logger.info('')
            logdir = logger.get_dir()
            if rank == 0 and logdir:
                if hasattr(env, 'get_state'):
                    with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                        pickle.dump(env.get_state(), f)
                if eval_env and hasattr(eval_env, 'get_state'):
                    with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                        pickle.dump(eval_env.get_state(), f)

            # ------------------------------ plot statistics every nb_epochs_unit -----------------------------------
            ddpg_rewards.append(np.mean(episode_rewards_history))
            if (epoch + 1) % nb_epochs_unit == 0:
                ddpg_suc_percents.append(suc_num / episode_num)
                # ---------- Evaluate for 5 iters -----------------------
                nb_eval_epochs = 5
                nb_eval_epoch_cycles = 5
                eval_episode_num = 0
                eval_suc_num = 0

                eval_episode_reward = 0
                eval_episode_step   = 0

                eval_epoch_episode_rewards = []
                eval_epoch_episode_steps   = []
                for i_epoch in range(nb_eval_epochs):
                    logger.log("********** Start Evaluation. Iteration %i ************" % i_epoch)
                    for i_cycle in range(nb_eval_epoch_cycles):
                        for t_rollout in range(nb_rollout_steps):
                            eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                            assert eval_action.shape == env.action_space.shape
                            eval_obs, eval_r, eval_done, eval_suc, eval_info = env.step(
                                max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])

                            eval_episode_reward += eval_r
                            eval_episode_step += 1
                            if eval_done:
                                eval_obs = env.reset()
                                eval_epoch_episode_rewards.append(eval_episode_reward)
                                eval_episode_rewards_history.append(eval_episode_reward)
                                eval_epoch_episode_steps.append(eval_episode_step)
                                eval_episode_reward = 0
                                eval_episode_step = 0

                                eval_episode_num += 1
                                if eval_suc:
                                    eval_suc_num += 1
                    logger.record_tabular("Eval_EpRewMean", np.mean(eval_episode_rewards_history))
                    logger.record_tabular("Eval_EpNumUntilNow", eval_episode_num)
                    logger.record_tabular("Eval_EpNumSuc", eval_suc_num)
                    logger.record_tabular("Eval_EpSucPercent", eval_suc_num / eval_episode_num)
                    logger.dump_tabular()
                    eval_ddpg_rewards.append(np.mean(eval_episode_rewards_history))
                eval_suc_percents.append(eval_suc_num / eval_episode_num)
                # ----------------------------------------------------------------------------------------------
                # --------------------- plotting and saving -------------------------
                if saver is not None:
                    logger.info("saving the trained model")
                    start_time_save = time.time()
                    if epoch + 1 == nb_epochs:
                        saver.save(sess, kwargs['MODEL_DIR'] + "/trained_model")
                    else:
                        saver.save(sess, kwargs['MODEL_DIR'] + "/iter_" + str((epoch + 1) // nb_epochs_unit))

                plot_performance(range(len(ddpg_rewards)), ddpg_rewards, ylabel=r'avg reward per DDPG learning step',
                                 xlabel='ddpg iteration', figfile=os.path.join(kwargs['FIGURE_DIR'], 'ddpg_reward'), title='TRAIN')
                plot_performance(range(len(ddpg_suc_percents)), ddpg_suc_percents,
                                 ylabel=r'overall success percentage per algorithm step under DDPG',
                                 xlabel='algorithm iteration', figfile=os.path.join(kwargs['FIGURE_DIR'], 'success_percent'),
                                 title="TRAIN")

                plot_performance(range(len(eval_ddpg_rewards)), eval_ddpg_rewards, ylabel=r'avg reward per DDPG eval step',
                                 xlabel='ddpg iteration', figfile=os.path.join(kwargs['FIGURE_DIR'], 'eval_ddpg_reward'),
                                 title='EVAL')
                plot_performance(range(len(eval_suc_percents)), eval_suc_percents,
                                 ylabel=r'overall eval success percentage per algorithm step under DDPG',
                                 xlabel='algorithm iteration', figfile=os.path.join(kwargs['FIGURE_DIR'], 'eval_success_percent'),
                                 title="EVAL")

                # save data which is accumulated UNTIL iter i
                with open(kwargs['RESULT_DIR'] + '/ddpg_reward_' + 'iter_' + str((epoch + 1) // nb_epochs_unit) + '.pickle', 'wb') as f2:
                    pickle.dump(ddpg_rewards, f2)
                with open(kwargs['RESULT_DIR'] + '/success_percent_' + 'iter_' + str((epoch + 1) // nb_epochs_unit) + '.pickle', 'wb') as fs:
                    pickle.dump(ddpg_suc_percents, fs)


                # save evaluation data accumulated until iter i
                with open(kwargs['RESULT_DIR'] + '/eval_ddpg_reward_' + 'iter_' + str((epoch + 1) // nb_epochs_unit) + '.pickle', 'wb') as f_er:
                    pickle.dump(eval_ddpg_rewards, f_er)
                with open(kwargs['RESULT_DIR'] + '/eval_success_percent_' + 'iter_' + str((epoch + 1) // nb_epochs_unit) + '.pickle', 'wb') as f_es:
                    pickle.dump(eval_suc_percents, f_es)
                # -------------------------------------------------------------------
            # -----------------------------------------------------------------------------------------------------




