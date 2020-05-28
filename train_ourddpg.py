# added by XLV: This file is modified for our project based on the main.py from TD3 folder: "/TD3/main.py"

import numpy as np
import torch
import gym
import argparse
import os
import time
from collections import *
import pandas as pd

from TD3 import utils
from TD3 import OurDDPG
from TD3 import DDPG
from TD3 import TD3

from gym_foo import gym_foo
from utils import logger
from utils.tools import *
import config

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=50):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	# added by XLV: compute number of successful episodes
	suc_episode_num = 0
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			# print("state: {}".format(state))
			action = policy.select_action(np.array(state))
			state, reward, done, info = eval_env.step(action)
			avg_reward += reward
			if env_name in ['DubinsCarEnv-v0', 'PlanarQuadEnv-v0'] and info['suc']:
				suc_episode_num += 1
	avg_reward /= eval_episodes
	suc_rate = suc_episode_num * 1.0 / eval_episodes
	logger.log("---------------------------------------")
	logger.log("Evaluation over {} episodes: avg reward {} success rate {}".format(eval_episodes, avg_reward, suc_rate))
	logger.log("---------------------------------------")
	return [avg_reward, suc_rate]


if __name__ == "__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="OurDDPG")               # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="PlanarQuadEnv-v0")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)               # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)    # Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)        # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=300e3, type=int)  # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                 # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)       # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                  # Discount factor
	parser.add_argument("--tau", default=0.005)                      # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)               # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                 # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)        # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")         # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                  # Model load file name, "" doesn't load, "default" uses file_name

	# added by XLV: more options for initialize Q network
	parser.add_argument("--initQ", default="no", type=str)
	parser.add_argument("--fixed", default="yes", type=str)
	parser.add_argument("--useGD", action="store_true")
	parser.add_argument("--save_debug_info", action="store_true")
	args = parser.parse_args()

	if args.initQ == "yes":
		args.initQ = True
		if args.fixed == "yes":
			args.fixed = True
			logger.log("You initialize Q target and want to keep it always fixed, thus Q target is never updated ...")
		else:
			args.fixed = False
			logger.log("You are initializing Q and want to update it accordingly ...")
	else:
		args.initQ = False
		args.fixed = False
		logger.log("not initializing Q, update Q target every train call ...")

	if args.useGD:
		args.initQ = False
		args.fixed = True

	# added by XLV: set a global pandas dataframe to save running statistics for further debug
	if args.env == "DubinsCarEnv-v0":
		config.debug_info = pd.DataFrame(columns=['QtargPred','reward', 'a1', 'a2', 'x','y','theta','d1','d2','d3','d4','d5','d6','d7','d8','nx','ny','ntheta','nd1','nd2','nd3','nd4','nd5','nd6','nd7','nd8'])
	elif args.env == "PlanarQuadEnv-v0":
		config.debug_info = pd.DataFrame(columns=['QtargPred', 'reward', 'a1', 'a2', 'x', 'vx', 'z', 'vz', 'theta', 'w', 'd1', 'd2', 'd3',
							  'd4', 'd5', 'd6', 'd7', 'd8', 'nx', 'nvx', 'nz', 'nvz', 'ntheta', 'nw', 'nd1', 'nd2', 'nd3', 'nd4', 'nd5', 'nd6', 'nd7','nd8'])


	file_name = "{}_{}_{}_{}_{}_{}".format(time.strftime('%d-%b-%Y_%H-%M-%S'), args.policy, args.env, args.seed, "yes" if args.initQ else "no", "fixed" if args.fixed else "non-fixed")
	file_name = file_name + "_{}".format("gd") if args.useGD else file_name

	config.RUN_ROOT = RUN_DIR = os.path.join("runs_log_ddpg", file_name)
	MODEL_DIR = os.path.join(RUN_DIR, 'model')
	FIGURE_DIR = os.path.join(RUN_DIR, 'figure')
	RESULT_DIR = os.path.join(RUN_DIR, 'result')
	if not os.path.exists(RUN_DIR):
		os.makedirs(RUN_DIR)
		os.makedirs(MODEL_DIR)
		os.makedirs(FIGURE_DIR)
		os.makedirs(RESULT_DIR)

	# logger initialize and configuration
	logger.configure(dir=RUN_DIR)

	logger.log("---------------------------------------")
	logger.log("Policy: {}, Env: {}, Seed: {}, Qinit: {}, Fixed: {}, UseGD: {}, Save debug info: {}".format(args.policy, args.env, args.seed, args.initQ, args.fixed, args.useGD, args.save_debug_info))
	logger.log("---------------------------------------")

	env = gym.make(args.env)
	policy = None

	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,

		# added by XLV: add a few more new kwargs
		"initQ": args.initQ,
		"fixed": args.fixed,
		"useGD": args.useGD,
		"save_debug_info": args.save_debug_info
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)
	elif args.policy == "OurDDPG":
		policy = OurDDPG.DDPG(**kwargs)
	elif args.policy == "DDPG":
		policy = DDPG.DDPG(**kwargs)
	else:
		raise ValueError("policy type is invalid")

	if args.load_model != "" and policy is not None:
		policy.load(args.load_model)

	# Initialize replay buffer
	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

	# Training statistics
	rewbuffer = deque(maxlen=100)
	total_train_rews = []
	
	# Evaluate untrained policy
	logger.log("start evaluating untrained policy ...")
	evaluations = [eval_policy(policy, args.env, args.seed)]
	logger.log("evaluating untrained policy finished ...")

	# Segment to hold consecutive episodes
	seg = {'obs':[], 'rews':[], 'news':[]}

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			print("we are randomly sampling action ...")
			action = env.action_space.sample()
		else:
			# print("state: {}".format(state))
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)


		# Perform action
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		# TODO
		# Store consecutive episodes (up to 1000 steps)
		# seg['obs'].append(state)
		# seg['next_obs'].append(next_state)
		# seg['rews'].append(reward)
		# seg['news'].append(done_bool)
		# if len(seg['rews']) == 1000:
		# 	news = seg['news'].append(0).deepcopy()
		# 	rews = seg['rews'].deepcopy()
		# 	G    = seg['rews'].append(0).deepcopy()
		# 	next_state = seg['next_obs'].deepcopy()
		#
		# 	target_Q = policy.critic_target(torch.FloatTensor(next_state).to(device), policy.actor_target(torch.FloatTensor(next_state)))
		# 	target_Q = reward + (news * args.discount * target_Q).detach()
		# 	for t in reversed(range(1000)):
		# 		nonterminal = 1 - news[t + 1]
		# 		G[t] = seg['rews'][t] + args.discount * G[t + 1] * nonterminal


		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		# if t >= args.start_timesteps and t % 50 == 0:
		# 	policy.train(replay_buffer, args.batch_size)
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			logger.log("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(t+1, episode_num+1, episode_timesteps, episode_reward))
			# added by XLV: store episodic reward to a buffer for rolling-average
			rewbuffer.append(episode_reward)
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Collect training reward every 1e3 steps
		if (t + 1) % 1e3 == 0:
			total_train_rews.append(np.mean(rewbuffer))

		# Evaluate episode and save and plot statistics
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			np.save(RESULT_DIR + "/{}".format("eval_result"), evaluations)  # eval_result: [(eval_rew, eval_suc_rate), ...]
			if args.save_model: policy.save(MODEL_DIR + "/{}".format("ddpg_model"))
			np.save(RESULT_DIR + "/{}".format("train_result"), total_train_rews)  # train_result: [train_rew, ...]

			# plot training reward performance
			train_plot_x = np.arange(len(total_train_rews)) + 1
			train_plot_x = np.insert(train_plot_x, 0, 0)
			train_plot_y = np.insert(total_train_rews, 0, total_train_rews[0])
			plot_performance(x=train_plot_x, y=train_plot_y, ylabel=r'training reward',
							 xlabel='ddpg timesteps (*1k)', figfile=os.path.join(FIGURE_DIR, 'train_reward'),
							 title='TRAIN')

			# plot evaluation rews and success rate
			eval_plot_x = (np.arange(len(evaluations)) + 1)
			eval_plot_x = np.insert(eval_plot_x, 0, 0)
			evaluations_rew = [eval_item[0] for eval_item in evaluations]
			evaluations_suc = [eval_item[1] for eval_item in evaluations]
			eval_plot_y = np.insert(evaluations_suc, 0, 0)
			plot_performance(x=eval_plot_x, y=eval_plot_y,
							 ylabel=r'eval success rate',
							 xlabel='ddpg timesteps (*{}k)'.format(args.eval_freq), figfile=os.path.join(FIGURE_DIR, 'eval_success_rate'),
							 title="EVAL")
			eval_plot_y = np.insert(evaluations_rew, 0, evaluations_rew[0])
			plot_performance(x=eval_plot_x, y=eval_plot_y,
							 ylabel=r'eval avg reward',
							 xlabel='ddpg timesteps (*{}k)'.format(args.eval_freq), figfile=os.path.join(FIGURE_DIR, 'eval_avg_reward'),
							 title="EVAL")
