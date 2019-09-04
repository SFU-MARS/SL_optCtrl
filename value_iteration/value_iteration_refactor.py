import math
import numpy as np
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import RegularGridInterpolator
import time

class env_quadrotor_6d(object):
	def __init__(self):
		########### World Setting ########## 
		self.x = (-5, 5)
		self.z = (0, 10)
		self.gravity = 9.8
		self.obstacles = None

		########### Drone Setting ##########
		self.vx = (-2, 2)
		self.vz = (-2, 2)
		self.theta = (-math.pi, math.pi)
		self.omega = (-2*math.pi, 2*math.pi)
		self.ranges = np.array([self.x, self.z, self.vx, self.vz, self.theta, self.omega])

		self.mass = 1.25
		self.length = 0.5
		self.inertia = 0.125
		self.trans = 0.25 # Cdv
		self.rot = 0.02255 # Cd_phi
		self.delta = 0.4

		self.t1 = (0, 36.7875 / 2)
		self.t2 = (0, 36.7875 / 2)
		self.actions = np.array([self.t1, self.t2])
		########### Goal Setting ##########
		self.goal_x = (3.5, 4.5)
		self.goal_z = (8.5, 9.5)
		self.goal_vx = (-2, 2)
		self.goal_vz = (-2, 2)
		self.goal_theta = (0.45, 1.05)
		self.goal_omega = self.omega
		self.goals = np.array([self.goal_x, self.goal_z, self.goal_vx, self.vz, self.theta, self.omega])

		########### Algorithm Setting ##########
		self.discount = 0.90
		self.threshold = 0.5

		# reward = [regula state, in goal, crashed, overspeed]
		self.reward = np.array([0, 1000, -400, -200], dtype = float)

		########## Discreteness Setting ##########
		# 6D state
		# 2D action
		# (x, z, vx, vz, theta, omega)
		# (t1, t2)
		self.state_step_num = np.array([11, 11, 11, 11, 9, 9])
		self.action_step_num = np.array([5, 5])

		########## Algorithm Declaration ##########
		self.state_grid = None
		self.action_grid = None

		self.value_x = None
		self.value_z = None

		self.reward_x = None
		self.reward_z = None

		self.state_type = None
		self.state_reward = None

		########## Subsystem Setting ##########
		self.dim = [[0, 2, 4, 5], [1, 3, 4, 5]]

	def add_obstacle(self, x1, x2, z1, z2):
		if ((x1 > x2) or (z1 > z2)):
			print("Add obstacle error! Check parameters")
			return

		if (self.obstacles is None):
			self.obstacles = np.array([[x1, x2, z1, z2]], dtype = float)
		else:
			self.obstacles = np.concatenate((self.obstacles, np.array([[x1, x2, z1, z2]])), axis = 0)

	def state_discretization(self):
		state_dim_num = len(self.state_step_num)
		action_dim_num = len(self.action_step_num)

		########### Space Declaration ##########
		l = []
		for size in self.state_step_num:
			x = np.empty((size), dtype = float)
			l.append(x)
		self.state_grid = np.array(l, dtype = object)

		l = []
		for size in self.action_step_num:
			x = np.empty((size), dtype = float)
			l.append(x)
		self.action_grid = np.array(l, dtype = object)

		########## Discretization Procedure ##########
		for i in range(state_dim_num):
			self.state_grid[i] = np.linspace(self.ranges[i][0], self.ranges[i][1], self.state_step_num[i])
		for i in range(action_dim_num):
			self.action_grid[i] = np.linspace(self.actions[i][0], self.actions[i][1], self.action_step_num[i])

	def state_init(self, system):
		if (system == "x"):
			self.value_x = np.zeros(self.state_step_num[self.dim_x], dtype = float)
			self.reward_x = np.zeros(self.state_step_num[self.dim_x], dtype = float)
			self.states = np.array(np.meshgrid(self.grid[0],
												self.grid[2],
												self.grid[4],
												self.grid[5])).T.reshape(-1, len(self.dim_x))

		else:
			self.value_z = np.zeros(self.state_step_num[self.dim_z], dtype = float)
			self.reward_z = np.zeros(self.state_step_num[self.dim_z], dtype = float)
			self.states = np.array(np.meshgrid(self.grid[1],
												self.grid[3],
												self.grid[4],
												self.grid[5])).T.reshape(-1, len(self.dim_y))

		self.reward = np.zeros(states.shape[0])
		delete_list = []
		for i, s in enumerate(states):
			state_tpe = self.state_check(s, self.d)


	def state_transition_x(self, state, action):
		action_sum = np.sum(action)
		action_diff = action[0] - action[1]

		state_ = np.array([state[0] + state[1] * self.delta,
							state[1] + self.delta / self.mass * (-self.trans * state[1] + math.sin(state[2]) * action_sum),
							state[2] + state[3] * self.delta,
							state[3] + self.delta / self.inertia * (-self.rot * state[3] + self.length * action_diff)])

		while (state_[2] > self.theta[1]):
			state_[2] = self.theta[0] + (state_[2] - self.theta[1])

		while (state_[2] < self.theta[0]):
			state_[2] = self.theta[1] + (state_[2] - self.theta[0])

		return state_

	def state_transition_y(self, state, action):
		action_sum = np.sum(action)
		action_diff = action[0] - action[1]

		state_ = np.array([state[0] + state[1] * self.delta,
							state[1] + self.delta * ((-self.trans * state[1] + math.cos(state[2]) * action_sum) / self.mass - self.gravity),
							state[2] + state[3] * self.delta,
							state[3] + self.delta / self.inertia * (-self.rot * state[3] + self.length * action_diff)
							])	

		while (state_[2] > self.theta[1]):
			state_[2] = self.theta[0] + (state_[2] - self.theta[1])

		while (state_[2] < self.theta[0]):
			state_[2] = self.theta[1] + (state_[2] - self.theta[0])

		return state_


env = env_quadrotor_6d()










