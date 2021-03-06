import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from keras.models import load_model

import pickle
import pandas as pd
import os

import config
from utils import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))



class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, initQ=False):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)

		# added by XLV: Q initialization configuration for Critic network
		if initQ:
			if state_dim == 11:
				self.user_config = {
					'Qweights_loadpath': '/local-scratch/xlv/SL_optCtrl/Qinit/dubinsCar/trained_model/256*256/Qf_weights.pkl'}
			elif state_dim == 14:
				self.user_config = {
					'Qweights_loadpath': '/local-scratch/xlv/SL_optCtrl/Qinit/PlanarQuad/trained_model/256*256/Qf_weights.pkl'}
			with open(self.user_config['Qweights_loadpath'], 'rb') as wt_f:
				print("start re-initialize Q function from {}".format(self.user_config['Qweights_loadpath']))
				wt = pickle.load(wt_f)

				assert np.transpose(wt[0][0]).shape == self.l1.weight.detach().numpy().shape
				self.l1.weight.data = torch.from_numpy(np.transpose(wt[0][0]))
				self.l1.bias.data = torch.from_numpy(wt[0][1])
				self.l4.weight.data = torch.from_numpy(np.transpose(wt[0][0]))
				self.l4.bias.data = torch.from_numpy(wt[0][1])

				assert np.transpose(wt[1][0]).shape == self.l2.weight.detach().numpy().shape
				self.l2.weight.data = torch.from_numpy(np.transpose(wt[1][0]))
				self.l2.bias.data = torch.from_numpy(wt[1][1])
				self.l5.weight.data = torch.from_numpy(np.transpose(wt[1][0]))
				self.l5.bias.data = torch.from_numpy(wt[1][1])

				assert np.transpose(wt[2][0]).shape == self.l3.weight.detach().numpy().shape
				self.l3.weight.data = torch.from_numpy(np.transpose(wt[2][0]))
				self.l3.bias.data = torch.from_numpy(wt[2][1])
				self.l6.weight.data = torch.from_numpy(np.transpose(wt[2][0]))
				self.l6.bias.data = torch.from_numpy(wt[2][1])
				print("weight re-initialize succeeds!")

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		initQ=False,
		fixed=True,
		useGD=False,
		useValInterp=False,
		save_debug_info=False
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		# added by XLV: add initQ as a new argument
		self.critic = Critic(state_dim, action_dim, initQ=False).to(device)
		self.critic_target = copy.deepcopy(self.critic) if not initQ else Critic(state_dim, action_dim, initQ=initQ).to(device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		# added by XLV: add a few new arguments
		self.state_dim = state_dim
		self.initQ = initQ
		self.fixed = fixed
		self.useGD = useGD
		self.useValInterp = useValInterp
		self.save_debug_info = save_debug_info

		# assert these two variables are not set as true at same time
		assert not (self.useGD and self.useValInterp)

		# Use Q ground truth to check the performance
		if self.useGD:
			if self.state_dim == 11:
				from value_iteration.value_iteration_3d.value_iteration_car_3d import env_dubin_car_3d
				valM = np.load("/local-scratch/xlv/SL_optCtrl/value_iteration/value_boltzmann_angle.npy")
				self.subenv = env_dubin_car_3d()
				self.subenv.algorithm_init()
				self.interp = self.subenv.set_interpolation(valM)
			elif self.state_dim == 14:
				from value_iteration.helper_function import value_interpolation_function_quad
				# valM_path = "/local-scratch/xlv/SL_optCtrl/value_iteration/value_iteration_6d_xubo_version_1/value_matrix_quad_6D/transfered_value_matrix_7.npy"
				# valM_path = "/local-scratch/xlv/SL_optCtrl/value_iteration/value_iteration_6d_xubo_version_1/value_matrix_quad_6D_boltzmann/transferred_value_matrix_8.npy"
				# valM_path = "/local-scratch/xlv/SL_optCtrl/value_iteration/value_iteration_6d_xubo_version_1/value_matrix_quad_6D_boltzmann_airspace_201910_ddpg/transferred_value_matrix_8.npy"
				# valM_path = "/local-scratch/xlv/SL_optCtrl/value_iteration/value_iteration_6d_xubo_version_1/value_matrix_quad_6D_boltzmann_fast_airspace_201910_ddpg/transferred_value_matrix_9.npy"
				valM_path = "/local-scratch/xlv/SL_optCtrl/value_iteration/value_iteration_6d_xubo_version_1/value_matrix_quad_6D_boltzmann_fast_airspace_201910_ddpg/trial_3/transferred_value_matrix_9.npy"
				self.subenv = value_interpolation_function_quad(valM_path)
				self.interp = self.subenv.setup()
			logger.log("You are using useGD, the Q target is calculated via interpolation ...")
			logger.log("The useGD comes from the file: {}".format(valM_path))

		if self.useValInterp:
			assert self.state_dim == 14
			# valinterp_path = "/local-scratch/xlv/SL_optCtrl/Qinit/quad/trained_model/vnn_interp/nn_interp.h5"  # vnn with 6d input
			# valinterp_path = "/local-scratch/xlv/SL_optCtrl/Qinit/quad/trained_model/qnn_interp/nn_interp.h5"  # qnn with 6d + 2d input
			# self.valinterp = load_model(valinterp_path)
			# logger.log("You are using useValInterp, the Q target is calculated via interpolation ...")
			# logger.log("The useValInterp comes from the file: {}".format(valinterp_path))

			valinterp_path = os.environ['PROJ_HOME_3'] + '/tf_model/quad/vf_vi.h5'  # vnn with 14d input (the one we use for PPO)
			self.valinterp = load_model(valinterp_path)		
			# apply normalization
			normbasedata_path = os.environ['PROJ_HOME_3'] + '/data/quad/valFunc_vi_filled_cleaned.csv'
			normbase_colnames = ['x', 'vx', 'z', 'vz', 'phi', 'w', 'value', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
			normbase_df = pd.read_csv(normbasedata_path, names=normbase_colnames, na_values="?", comment='\t', sep=",",
									skipinitialspace=True, skiprows=1)
			stats_source = normbase_df.copy()
			stats_source.dropna()
			stats_source.pop("value")
			stats = stats_source.describe()
			stats = stats.transpose()
			self.stats_mean = stats['mean'].to_numpy().reshape(1, -1)
			self.stats_std  = stats['std'].to_numpy().reshape(1, -1)
			logger.log("You are using useValInterp, the Q target is calculated via interpolation ...")
			logger.log("The useValInterp comes from the file: {}".format(valinterp_path))


		self.total_it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=100, updateQ=False):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		# Compute the target Q value
		target_Q = None
		if not self.useGD and not self.useValInterp:
			with torch.no_grad():
				# Select action according to policy and add clipped noise
				noise = (
					torch.randn_like(action) * self.policy_noise
				).clamp(-self.noise_clip, self.noise_clip)

				next_action = (
					self.actor_target(next_state) + noise
				).clamp(-self.max_action, self.max_action)

				# Compute the target Q value
				target_Q1, target_Q2 = self.critic_target(next_state, next_action)
				target_Q = torch.min(target_Q1, target_Q2)
				target_Q = reward + not_done * self.discount * target_Q
		
		elif self.useGD:
			if self.state_dim == 11:
				val_s_prime = self.interp(next_state[:, :3])
				val_s_prime = val_s_prime.reshape(-1,1)
				target_Q    = reward + not_done * self.discount * torch.FloatTensor(val_s_prime)
			elif self.state_dim == 14:
				# check to use transferred matrix
				val_s_prime = self.interp(next_state[:, :6])
				val_s_prime = val_s_prime.reshape(-1, 1)
				target_Q    = reward + not_done * self.discount * torch.FloatTensor(val_s_prime)
		
		elif self.useValInterp:
			# Uncomment this if you use vnn with 6d input (performance is not good)
			# input = next_state.detach().numpy()[:, :6]   
			# val_s_prime = self.valinterp.predict(input)
			# val_s_prime = val_s_prime.reshape(-1, 1)
			# target_Q	= reward + not_done * self.discount * torch.FloatTensor(val_s_prime)
			

			# Uncomment this if you use vnn with 14d input (the one we use on PPO)
			input = next_state.detach().numpy()
			input = (input - self.stats_mean) / self.stats_std
			val_s_prime = self.valinterp.predict(input)
			val_s_prime = val_s_prime.reshape(-1, 1)
			target_Q	= reward + not_done * self.discount * torch.FloatTensor(val_s_prime)


			# Uncomment here and the following if you use qnn to estimate target_Q
			# input_s = state.detach().numpy()[:, :6]   
			# input_a = action.detach().numpy()
			# input_a = np.clip(input_a, -2, 2)
			# input_a = 7 + (10 - 7) * (input_a - (-2)) / (2 - (-2))
			# input_sa = np.concatenate((input_s, input_a), axis=1)
			# target_Q = self.valinterp.predict(input_sa)
			# target_Q = torch.FloatTensor(target_Q)
		else:
			raise ValueError("invalid setting !")

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models (added by XLV: temporarily distable critic target update)
			if not self.initQ:
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			else:
				if not self.fixed:
					if updateQ:
						for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
							target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		# Save some statistics for further checking
		if self.save_debug_info:
			QtargPred, s, a, ns, r = target_Q.detach().numpy(), state.detach().numpy(), action.detach().numpy(), next_state.detach().numpy(), reward.detach().numpy()
			log_data = np.concatenate((QtargPred, r, a, s, ns), axis=1)

			# Only choose 1 sample from each batch to save
			index = np.random.choice(log_data.shape[0], 1, replace=False)
			log_data = log_data[index]
			pd_columns = None
			if self.state_dim == 11:
				pd_columns = ['QtargPred', 'reward', 'a1', 'a2', 'x', 'y', 'theta', 'd1', 'd2', 'd3', 'd4',
							  'd5', 'd6', 'd7', 'd8', 'nx', 'ny', 'ntheta', 'nd1', 'nd2', 'nd3', 'nd4', 'nd5', 'nd6', 'nd7','nd8']
			elif self.state_dim == 14:
				pd_columns = ['QtargPred', 'reward', 'a1', 'a2', 'x', 'vx', 'z', 'vz', 'theta', 'w', 'd1', 'd2',
							  'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'nx', 'nvx', 'nz', 'nvz', 'ntheta', 'nw', 'nd1',
							  'nd2', 'nd3', 'nd4', 'nd5', 'nd6', 'nd7', 'nd8']
			log_df = pd.DataFrame(log_data, columns=pd_columns)

			config.debug_info = config.debug_info.append(log_df, ignore_index=True)
			if len(config.debug_info) > 10e3:
				with open(config.RUN_ROOT + "/debug_info.csv", 'a') as f:
					config.debug_info.to_csv(f, header=True)
					config.debug_info = pd.DataFrame(columns=pd_columns)
					print("debug info saving good!")

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)