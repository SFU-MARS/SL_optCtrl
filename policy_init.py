import numpy as np
import tensorflow as tf  # first load tensorflow and then load pytorch
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


if __name__ == "__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	######################################
	###### global arguments setting ######

	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")          	     		# Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="DubinsCarEnv-v0") 	     		# OpenAI gym environment name
	parser.add_argument("--batch_size", default=256, type=int)       		# Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                  		# Discount factor
	parser.add_argument("--policy_noise", default=0.2)               		# Noise added to target policy during critic update
	parser.add_argument("--tau", default=0.005)                      		# Target network update rate
	parser.add_argument("--noise_clip", default=0.5)                 		# Range to clip target policy noise
	parser.add_argument("--max_timesteps", default = 3e6, type = int)		# Total timesteps in training stage
	parser.add_argument("--warm_start_timesteps", default = 1e4, type = int)# Timesteps in warm start stage (iteration 0)
	parser.add_argument("--expl_noise", default=0.1)               		    # Std of Gaussian exploration noise
	parser.add_argument("--timesteps_to_update", default = 10, type = int)# Update value and policy neural network every N timesteps
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

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

	#########################################################


	##################################
	###### Initialize variables ######
	current_episode = 0
	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	###############

	###### warm start to collect trajectories for buffer ######
	logger.log("===============================================")
	logger.log("warm start begins !!! Total timesteps: {}".format(args.warm_start_timesteps))			

	for t in range(int(args.warm_start_timesteps)):

		# select an action from stochastic policy
		action = env.action_space.sample()
		
		# Do one step in environment and store data in replay buffer
		next_state, reward, done, _ = env.step(action)
		episode_timesteps += 1 
		done_bool = float(done)

		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Reset the environment if one episode is done, and log data
		if (done  or  t == args.warm_start_timesteps): 
			logger.log("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(t+1, episode_num+1, episode_timesteps, episode_reward))			
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 


	logger.log("warm start ends !!!")
	logger.log("===============================================")
	########################


	################################
	##### Initialize varibles ######
	current_episode = 0
	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	update_num = 0  
	##############

	#################################
	###### Main training stage ######
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




