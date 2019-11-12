import math
import numpy as np
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import RegularGridInterpolator
import time

class env_quadrotor_r6d(object):
	def __init__(self):
		########## World Setting ##########
		self.x = (-5, 5)
		self.z = (0, 10)
		self.gravity = 9.81
		self.obstacles = None

		########## Drone Setting ##########
		self.vx = (-2, 2)
		self.vz = (-2, 2)
		self.theta = (-math.pi, math.pi)
		self.omega = (-2*math.pi, 2*math.pi)
		self.ranges = np.array([self.x, self.z, self.theta, self.vx, self.vz, self.omega])

		self.mass = 1.27
		self.length = 0.4
		self.inertia = 0.125
		self.trans = 0.25
		self.rot = 0.02255
		self.delta = 0.4

		self.t1 = (0, 12)
		self.t2 = (0, 12)
		self.actions = np.array([self.t1, self.t2])

		########## Goal Setting ##########
		self.goal_x = (3, 5)
		self.goal_z = (8, 10)
		self.goal_vx = self.vx
		self.goal_vz = self.vz
		self.goal_theta = self.theta
		self.goal_omega = self.omega
		self.goals = np.array([self.goal_x, self.goal_z, self.goal_theta, self.goal_vx, self.goal_vz, self.goal_omega])

		########## Algorithm Setting ##########
		self.discount = 0.9
		self.threshold = 2

		# reward = [regular state, in goal, crashed, overspeed]
		self.reward_list = np.array([0, 1000, -400, -200], dtype = float)

		########## Discreteness Setting ##########
		self.state_step_num = np.array([11, 11, 11, 11, 9, 9])
		self.action_step_num = np.array([5, 5])

		########## Algorithm Declaration ##########
		self.state_grid = None
		self.action_grid = None

		self.value = None
		self.reward = None

		self.state_type = None
		self.state_reward = None

	def add_obstacle(self, x1, x2, z1, z2):
		if ((x1 > x2) or (z1 > z2)):
			print("Add obstacle error! Check parrameters")
			return

		if (self.obstacles is None):
			self.obstacles = np.array([[x1, x2, z1, z2]], dtype = float)
		else:
			self.obstacles = np.concatenate((self.obstacles, np.array([[x1, x2, z1, z2]])), axis = 0)

	def algorithm_init(self):
		self.state_discretization()
		self.state_init()
		# self.action_init()
		# self.add_obstacles()

	def state_discretization(self):
		state_dim_num = len(self.state_step_num)
		action_dim_num = len(self.action_step_num)

		########## Space Declaration ##########
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

	def state_init(self):
		self.states = np.array(np.meshgrid(self.state_grid[0],
											self.state_grid[1],
											self.state_grid[2],
											self.state_grid[3],
											self.state_grid[4],
											self.state_grid[5])).T.reshape(-1, len(self.state_grid))

		self.value = np.zeros(self.state_step_num, dtype = float)
		self.reward = np.zeros(self.states.shape[0], dtype = float)
		delete_list = []

		for i, s in enumerate(self.states):
			state_type = self.state_check(s)
			if (state_type == 1  or  state_type == 2):
				index = tuple(self.state_to_index(s))
				self.value[index] = self.reward_list[state_type]
				delete_list.append(i)

	def action_init(self):
		self.acts = np.array(np.meshgrid(self.action_grid[0],
										self.action_grid[1])).T.reshape(-1, self.actions.shape[0])

	def value_iteration(self):
		iteration = 0
		while True:
			delta = 0
			for i, s in enumerate(self.states):
				best_value = -10000000
				state_type = self.state_check(s)
				if (state_type > 0):
					continue

				current_reward = self.reward[i]

				for i, a in enumerate(self.acts):
					s_ = self.state_transition(s, a)

					next_step_type = self.state_check(s_)
					if (next_step_type >= 2):
						next_step_value = self.reward_list[next_step_type]
					else:
						sub_value_matrix, sub_states = self.seek_neighbors_values(s_)
						interpolating_function = RegularGridInterpolator((sub_states[0],
																			sub_states[1],
																			sub_states[2],
																			sub_states[3],
																			sub_states[4],
																			sub_states[5]),
																			sub_value_matrix,
																			bounds_error = False,
																			fill_value = self.reward_list[2])

						next_step_value = interpolating_function(s_)

					index_s = self.state_to_index(s)
					index_a = self.action_to_index(a)
					index_all = self.combine_to_tuple(index_s, index_a)

					self.qvalue[index_all] = current_reward + self.discount * next_step_value
					best_value = max(best_value, current_reward + self.discount * next_step_value)

				index = tuple(self.state_to_index(s))
				current_delta = abs(best_value - self.value[index])
				delta = max(delta, current_delta)
				self.value[index] = best_value

			self.value_output(iteration, True)
			print("iteration %d:" %(iteration))
			print("delta: ", delta)
			print("\n\n")

			iteration += 1

			if (delta < self.threshold):
				break

	def state_transition(self, state, action):
		action_sum = np.sum(action)
		action_diff = action[0] - action[1]

		#TODO.....
		state_ = np.array([state[0] + state[1] * self.delta,
							state[1] + self.delta / self.mass * (-self.trans * state[1] + math.sin(state[2]) * action_sum),
							state[2] + state[3] * self.delta,
							state[3] + self.delta / self.inertia * (-self.rot * state[3] + self.length * action_diff)])

		#TODO.....


	def state_check(self, s):
		temp = 0
		temp = self.check_goal(s)
		if (temp):
			return temp

		temp = self.check_crash(s)
		if (temp):
			return temp

		temp = self.check_overspeed(s)
		if (temp):
			return temp

		return temp

	def check_goal(self, s):
		for i, v in enumerate(s):
			if (v < self.goals[i][0]  or  v > self.goals[i][1]):
				return 0

		return 1

	def check_crash(self, s, width, radius):
		# return 2 = crashed
		# return 0 = no crash

		if (self.obstacles is not None  and  len(self.obstacles)):
			for obs in self.obstacles:
				if (s[0] >= obs[0] - width - radius  and  
					s[0] <= obs[1] + width + radius  and
					s[1] >= obs[2] - width - radius  and  
					s[1] <= obs[3] + width + radius):
					return 2

		if (s[0] <= self.ranges[0][0] + radius  or  
			s[0] >= self.ranges[0][1] - radius):
			return 2

		if (s[1] < self.ranges[1][0] + radius  or  
			s[1] > self.ranges[1][1] - radius):
			return 2

		return 0 

	def check_overspeed(self, s):
		# return 3 = overspeed
		# return 0 = regular speed
		# Only check the velocity and angular velocity

		for i, v in enumerate(s):
			if (i < 3):
				continue

			if (v < self.ranges[i][0]  or  v > self.ranges[i][1]):
				return 3

		return 0
		
	def state_to_index(self, s):
		grid = np.copy(self.state_grid)

		index = []
		for i, v in enumerate(s):
			grid[i] = np.absolute(grid[i] - v)
			index.append(grid[i].argmin())

		return index

	def action_to_index(self, a):
		grid = np.copy(self.action_grid)

		index = []
		for i, v in enumerate(a):
			grid[i] = np.absolute(grid[i] - v)
			index.append(grid[i].argmin())

		return index




if __name__ == "__main__":
	env = env_quadrotor_r6d()
	env.algorithm_init()