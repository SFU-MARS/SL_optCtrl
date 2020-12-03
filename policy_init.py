import numpy as np
import tensorflow as tf  # first load tensorflow and then load pytorch
import torch
import gym
import argparse
import os
import time
from collections import *
import pandas as pd

from TD3_mbi import utils
from TD3_mbi import OurDDPG
from TD3_mbi import DDPG
from TD3_mbi import TD3
from TD3_mbi import OurDDPG_mbi

from gym_foo import gym_foo
from utils import logger
from utils.tools import *
import config


if __name__ == "__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	######################################
	###### global arguments setting ######

	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="OurDDPG")  	 				# Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="DubinsCarEnv-v0") 		 		# OpenAI gym environment name
	parser.add_argument("--batch_size", default=256, type=int) 	  			# Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)  						# Discount factor
	parser.add_argument("--policy_noise", default=0.2)   					# Noise added to target policy during critic update
	parser.add_argument("--tau", default=0.005)  							# Target network update rate
	parser.add_argument("--noise_clip", default=0.5) 						# Range to clip target policy noise
	parser.add_argument("--max_timesteps", default = 3e6, type = int)		# Total timesteps in training stage
	parser.add_argument("--warm_start_timesteps", default = 1e2, type = int)# Timesteps in warm start stage (iteration 0)
	parser.add_argument("--expl_noise", default=0.1)   						# Std of Gaussian exploration noise
	parser.add_argument("--timesteps_to_update", default = 10, type = int)	# Update value and policy neural network every N timesteps
	parser.add_argument("--actor_pretrain_epoch", default = 10000, type = int)# The epoch of actor pre-train by value iteration samples
	args = parser.parse_args()

	##########################

	##############################
	###### Create save path ######

	file_name = "{}_{}_{}".format(time.strftime('%d-%b-%Y_%H-%M-%S'), args.policy, args.env)

	config.RUN_ROOT = RUN_DIR = os.path.join("runs_log_policy_init", file_name)
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

	logger.log("===============================================")
	logger.log("Policy: {}".format(args.policy))
	logger.log("Env: {}".format(args.env))
	logger.log("Batch size: {}".format(args.batch_size))
	logger.log("Discount: {}".format(args.discount))
	logger.log("Policy noise: {}".format(args.policy_noise))
	logger.log("tau (target network update rate): {}".format(args.tau))
	logger.log("noise clip: {}".format(args.noise_clip))
	logger.log("max timesteps: {}".format(args.max_timesteps))
	logger.log("warm start timesteps: {}".format(args.warm_start_timesteps))
	logger.log("exploration noise: {}".format(args.expl_noise))
	logger.log("timesteps to update (How often updates networks): {}".format(args.timesteps_to_update))
	logger.log("The number of epoch for pre-train the actor model: {}".format(args.actor_pretrain_epoch))
	# logger.log(": {}".format())
	logger.log("===============================================")

	#####################################################

	#####################################################
	###### Initialize env and actor-critic network ######
	env = gym.make(args.env)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau
	}

	policy = DDPG.DDPG(**kwargs)
	# policy = OurDDPG_mbi.DDPG(**kwargs)

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	#########################################################


	##### Check Q-function initialization #####
	# state_dim = 11
	# action_dim = 2 
	# max_action = 2.0

	# kwargs = {
	# 	"state_dim": state_dim,
	# 	"action_dim": action_dim,
	# 	"max_action": max_action,
	# 	"discount": args.discount,
	# 	"tau": args.tau
	# }

	# policy = OurDDPG_mbi.DDPG(**kwargs)

	# replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

	# dataset_path = "./data/dubins/polFunc_vi_filled_cleaned.csv"
	# column_names = ['x', 'y', 'theta', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'vel', 'ang_vel', 'value']
	# raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=",", skipinitialspace=True, skiprows=1)
	# dataset = raw_dataset.copy()
	# dataset = dataset.dropna()
	# train_dataset = dataset.sample(frac=1.0, random_state=0)
	# stats = train_dataset.describe()
	# mean = stats.loc[['mean']].to_numpy()
	# std = stats.loc[['std']].to_numpy()
	# mean = mean.reshape(-1)
	# std = std.reshape(-1)
	# state_mean = mean[:11]
	# state_std = std[:11]
	# action_mean = mean[11:13]
	# action_std = std[11:13]
	# # print(mean)
	# # print(std)

	# # print(state_mean)
	# # print(state_std)
	# # print(action_mean)
	# # print(action_std)

	# state = [-4.2386, -3.6900,  0.1317,  1.1328,  1.4048,  2.3014,  5.2067,  5.4090, 8.3708,  6.0052,  8.6398]
	# # state = (state - state_mean) / state_std
	# state = torch.FloatTensor([state]).to(device)
	# action = [-2.0000,  1.9138]
	# # action = (action - action_mean) / action_std
	# action = torch.FloatTensor([action]).to(device)
	# # print(action)
	# print(state, action)
	# print(policy.critic(state, action).item())
	###################

	##### create stats for observation normalization #####
	# dataset_path = "./data/dubins/polFunc_vi_filled_cleaned.csv"
	# column_names = ['x', 'y', 'theta', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'vel', 'ang_vel', 'value']
	# raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=",", skipinitialspace=True, skiprows=1)
	# dataset = raw_dataset.copy()
	# dataset = dataset.dropna()
	# train_dataset = dataset.sample(frac=1.0, random_state=0)
	# stats = train_dataset.describe()
	# mean = stats.loc[['mean']].to_numpy()
	# std = stats.loc[['std']].to_numpy()
	# mean = mean.reshape(-1)
	# std = std.reshape(-1)
	# state_mean = mean[:11]
	# state_std = std[:11]
	# action_mean = mean[11:13]
	# action_std = std[11:13]
	####################### 



	##################################
	###### Initialize variables ######
	# current_episode = 0
	# state, done = env.reset(), False
	# episode_reward = 0
	# episode_timesteps = 0
	# episode_num = 0
	###############

	###### warm start to collect trajectories for buffer ######
	# logger.log("===============================================")
	# logger.log("warm start begins !!! Total timesteps: {}".format(args.warm_start_timesteps))			

	# for t in range(int(args.warm_start_timesteps)):

	# 	# select an action from stochastic policy
	# 	action = env.action_space.sample()
		
	# 	# Do one step in environment and store data in replay buffer
	# 	next_state, reward, done, _ = env.step(action)
	# 	episode_timesteps += 1 
	# 	done_bool = float(done)

	# 	replay_buffer.add(state, action, next_state, reward, done_bool)

	# 	state = next_state
	# 	episode_reward += reward

	# 	# Reset the environment if one episode is done, and log data
	# 	if (done  or  t == args.warm_start_timesteps): 
	# 		logger.log("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(t+1, episode_num+1, episode_timesteps, episode_reward))			
	# 		state, done = env.reset(), False
	# 		episode_reward = 0
	# 		episode_timesteps = 0
	# 		episode_num += 1 


	# logger.log("warm start ends !!!")
	# logger.log("===============================================")
	# ########################

	##### Add value iteration sample into replay buffer to pre-train actor network #####
	sample_file = "./data/dubins/polFunc_vi_filled_cleaned.csv"
	column_names = ['x', 'y', 'theta', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'vel', 'ang_vel', 'value']
	data = pd.read_csv(sample_file, names=column_names, na_values="?", comment='\t', sep=",", skipinitialspace=True, skiprows=1)
	data = data.dropna().to_numpy()
	state = data[:, :11]
	action = data[:, 11:13]
	for i, (s, a) in enumerate(zip(state, action)):
		if (data[i, -1] >= 0):
			# print(data[i], s, a)
			replay_buffer.add(s, a, s, 0.0, 0.0)

	# print(replay_buffer.size)

	# for i in range(args.actor_pretrain_epoch):
	# for i in range(10000):
	# 	if (i % 10 == 0):
	# 		logger.log("The {} pre-training begins......".format(i))
	# 	# policy.actor_train(replay_buffer, args.batch_size, i)
	# 	policy.actor_train(replay_buffer, 256, i)

	################################################

	##### Evaluation pre-train model #####
	# current_episode = 0
	# state, done = env.reset(), False
	# episode_reward = 0
	# episode_timesteps = 0
	# episode_num = 0

	# for t in range(int(args.max_timesteps)):
	# 	action = policy.select_action(np.array(state)).clip(-max_action, max_action)
	# 	next_state, reward, done, _ = env.step(action)
	# 	state = next_state
	# 	if (done):
	# 		state, done = env.reset(), False

	######################################


	# ################################
	# ##### Initialize varibles ######
	current_episode = 0
	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	update_num = 0  
	# ##############

	# #################################
	# ###### Main training stage ######
	logger.log("===============================================")
	logger.log("Main training begins !!! Total timesteps: {}".format(args.max_timesteps))
	for t in range(int(args.max_timesteps)):

		# select an action from stochastic policy
		action = (policy.select_action(np.array(state)) + 
				  np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)


		# Do one step in environment and store data in replay buffer
		next_state, reward, done, _ = env.step(action)
		episode_timesteps += 1
		done_bool = float(done)

		replay_buffer.add(state, action, next_state, reward, done_bool)


		state = next_state
		episode_reward += reward

		# Reset the environment if one episode is done, and log data
		if (done): 
			logger.log("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(t+1, episode_num+1, episode_timesteps, episode_reward))			
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

		if (t % int(args.timesteps_to_update) == 0):
			update_num += 1
			logger.log("The {} training begins......".format(update_num))
			policy.train(replay_buffer, args.batch_size)


	logger.log("Main training ends !!!")
	logger.log("===============================================")




