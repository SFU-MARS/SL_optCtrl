import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
import pandas as pd

import config
from utils import logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action

	
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, initQ=False):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, 1)

		# added by XLV: Q initialization configuration for Critic network
		if initQ:
			if state_dim == 11:
				self.user_config = {
					'Qweights_loadpath': '/local-scratch/xlv/SL_optCtrl/Qinit/dubinsCar/trained_model/400*300/Qf_weights.pkl'}
			elif state_dim == 14:
				self.user_config = {
					'Qweights_loadpath': '/local-scratch/xlv/SL_optCtrl/Qinit/PlanarQuad/trained_model/400*300/Qf_weights.pkl'}
			with open(self.user_config['Qweights_loadpath'], 'rb') as wt_f:
				print("start re-initialize Q function from {}".format(self.user_config['Qweights_loadpath']))
				wt = pickle.load(wt_f)
				assert np.transpose(wt[0][0]).shape == self.l1.weight.detach().numpy().shape
				self.l1.weight.data = torch.from_numpy(np.transpose(wt[0][0]))
				self.l1.bias.data = torch.from_numpy(wt[0][1])

				assert np.transpose(wt[1][0]).shape == self.l2.weight.detach().numpy().shape
				self.l2.weight.data = torch.from_numpy(np.transpose(wt[1][0]))
				self.l2.bias.data = torch.from_numpy(wt[1][1])

				assert np.transpose(wt[2][0]).shape == self.l3.weight.detach().numpy().shape
				self.l3.weight.data = torch.from_numpy(np.transpose(wt[2][0]))
				self.l3.bias.data = torch.from_numpy(wt[2][1])
				print("weight re-initialize succeeds!")


	def forward(self, state, action):
		q = F.relu(self.l1(torch.cat([state, action], 1)))
		q = F.relu(self.l2(q))
		return self.l3(q)


class DDPG(object):
	def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, initQ=False, fixed=True, useGD=False, save_debug_info=False):
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

		# added by XLV: add initQ as a new argument
		self.critic = Critic(state_dim, action_dim, initQ=False).to(device)
		self.critic_target = copy.deepcopy(self.critic) if not initQ else Critic(state_dim, action_dim, initQ=initQ).to(device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

		self.discount = discount
		self.tau = tau

		# added by XLV: add a few new arguments
		self.state_dim = state_dim
		self.initQ = initQ
		self.fixed = fixed
		self.useGD = useGD
		self.save_debug_info = save_debug_info

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
				valM_path = "/local-scratch/xlv/SL_optCtrl/value_iteration/value_iteration_6d_xubo_version_1/value_matrix_quad_6D_boltzmann_airspace_201910_ddpg/transferred_value_matrix_8.npy"
				self.subenv = value_interpolation_function_quad(valM_path)
				self.interp = self.subenv.setup()
			logger.log("You are using useGD, the Q target is calculated via interpolation ...")
			logger.log("The useGD comes from the file: {}".format(valM_path))


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=100, updateQ=False):
		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		# Compute the target Q value
		target_Q = None
		if not self.useGD:
			target_Q = self.critic_target(next_state, self.actor_target(next_state))
			target_Q = reward + (not_done * self.discount * target_Q).detach()
		else:
			if self.state_dim == 11:
				val_s_prime = self.interp(next_state[:, :3])
				val_s_prime = val_s_prime.reshape(-1,1)
				target_Q    = reward + not_done * self.discount * torch.FloatTensor(val_s_prime)
			elif self.state_dim == 14:
				# check to use transferred matrix
				val_s_prime = self.interp(next_state[:, :6])
				val_s_prime = val_s_prime.reshape(-1, 1)
				target_Q    = reward + not_done * self.discount * torch.FloatTensor(val_s_prime)

		# Get current Q estimate
		current_Q = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Compute actor loss
		actor_loss = -self.critic(state, self.actor(state)).mean()
		
		# Optimize the actor 
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Update the frozen target models (added by XLV: update critic target upon some condition)
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
				pd_columns = ['QtargPred', 'reward', 'a1', 'a2', 'x', 'y', 'theta', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6',
							  'd7', 'd8', 'nx', 'ny', 'ntheta', 'nd1', 'nd2', 'nd3', 'nd4', 'nd5', 'nd6', 'nd7', 'nd8']
			elif self.state_dim == 14:
				pd_columns = ['QtargPred', 'reward', 'a1', 'a2', 'x', 'vx', 'z', 'vz', 'theta', 'w', 'd1', 'd2', 'd3',
							  'd4', 'd5', 'd6', 'd7', 'd8', 'nx', 'nvx', 'nz', 'nvz', 'ntheta', 'nw', 'nd1', 'nd2', 'nd3', 'nd4', 'nd5', 'nd6', 'nd7','nd8']
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
		
		
if __name__ == "__main__":
	ac = Actor(11,2,2)
	print("layer1 weight shape:", ac.l1.weight.detach().numpy().shape)
	print("layer2 weight shape:", ac.l2.weight.detach().numpy().shape)
	print("layer3 weight shape:", ac.l3.weight.detach().numpy().shape)