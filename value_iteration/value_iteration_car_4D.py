import math
import numpy as np
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import RegularGridInterpolator
import time

class env_car_4d(object):
	def __init__(self):
		########### World Setting ########## 
		self.x = (-5, 5)
		self.z = (0, 10)
		self.obstacles = None

		########### Car Setting ##########
		self.x = (0, 10)
		self.y = (0, 10)
		self.theta = (-math.pi, math.pi)
		self.steer_angle = (-math.pi, math.pi)

		self.v = (-5, 5)
		self.omega = (-2*math.pi, 2*math.pi)

		self.ranges = np.array([self.x, self.y, self.theta, self.steer_angle])
		self.actions = np.array([self.v, self.omega])

		self.length = 0.3
		########### Goal Setting ##########
		self.goal_x = (8, 10)
		self.goal_z = (8, 10)
		self.goal_theta = self.theta
		self.goal_steer_angle = self.steer_angle
		self.goals = np.array([self.goal_x, self.goal_z, self.goal_theta, self.goal_steer_angle])

		########### Algorithm Setting ##########
		self.discount = 0.90
		self.threshold = 0.5
		self.delta = 0.2

		# reward = [regula state, in goal, crashed]
		self.reward_list = np.array([0, 1000, -400], dtype = float)

		########## Discreteness Setting ##########
		# 4D state
		# 2D action
		# (x, y, theta, steer_angle)
		# (v, omega)
		self.state_step_num = np.array([5, 5, 9, 5])
		self.action_step_num = np.array([5, 5])
		self.all_step_num = np.concatenate((self.state_step_num, self.action_step_num), axis = None)

		########## Algorithm Declaration ##########
		self.state_grid = None
		self.action_grid = None

		self.value = None
		self.reward = None

		self.state_type = None
		self.state_reward = None

		self.acts = None

	def algorithm_init(self):
		self.state_discretization()
		self.state_init()
		self.action_init()

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
											self.state_grid[3])).T.reshape(-1, len(self.state_grid))

		self.value = np.zeros(self.state_step_num, dtype = float)
		self.reward = np.zeros(self.states.shape[0], dtype = float)

		delete_list = []		

		for i, s in enumerate(self.states):
			state_type = self.state_check(s)
			if (state_type == 1  or  state_type == 2):
				index = tuple(self.state_to_index(s))
				self.value[index] = self.reward_list[state_type]
				self.reward[i] = self.reward_list[state_type]

		# print(self.states)


	def action_init(self):
		self.acts = np.array(np.meshgrid(self.action_grid[0],
										self.action_grid[1])).T.reshape(-1, self.actions.shape[0])
		# print(self.acts)

	def state_check(self, s):
		temp = 0
		temp = self.check_goal(s)
		if (temp):
			return temp

		temp = self.check_crash(s)
		if (temp):
			return temp

		return temp

	def check_goal(self, s):
		for i, v in enumerate(s):
			if (v < self.goals[i][0]  or  v > self.goals[i][1]):
				return 0

		return 1

	def check_crash(self, s):
		# return 2 = crashed
		# return 0 = no crash

		width = 0
		radius = 0

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
																			sub_states[3]),
																			sub_value_matrix,
																			bounds_error = False,
																			fill_value = self.reward_list[2])

						next_step_value = interpolating_function(s_)

					index_s = self.state_to_index(s)
					index_a = self.action_to_index(a)
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

	def value_output(self, iteration, readable_file = False):
		dir_path = "./value_matrix_car_4D/"
		try:
			os.mkdir(dir_path)
		except:
			print(dir_path + " exist!")

		file_name = "value_matrix" + "_" + str(iteration)
		print(file_name)

		np.save(dir_path + file_name, self.value)

		if (readable_file == True):
			file_name = file_name + ".txt"
			f = open(dir_path + file_name, "w")

			for i, s in enumerate(self.states):
				index = tuple(self.state_to_index(s))
				s = np.array2string(s, precision = 4, separator = '  ')
				s += '  ' + format(self.value[index], '.4f') + '\n'
				f.write(s)

			f.close()

	def seek_neighbors_values(self, state):
		index = self.state_to_index(state)
		r = []
		sub_states = []

		for i in range(len(index)):
			left = index[i]
			right = 0
			if (left == 0  or  left == self.state_step_num[i] - 1):
				right = 1 if left == 0 else left - 1
			else:
				if (self.state_grid[i][left + 1] - state[i]) < (state[i] - self.state_grid[i][left - 1]):
					right = left + 1
				else:
					right = left - 1
			left, right = left - (left > right), right + (left > right)
			right += 1
			r.append([left, right])
			sub_states.append(self.state_grid[i][left:right])

		return self.value[r[0][0]:r[0][1], 
							r[1][0]:r[1][1], 
							r[2][0]:r[2][1], 
							r[3][0]:r[3][1]], sub_states

	def state_transition(self, state, action):
		state_ = np.array([state[0] + self.delta * action[0] * math.cos(state[2] + state[3]),
							state[1] + self.delta * action[0] * math.sin(state[2] + state[3]),
							state[2] + (self.delta * action[0] * math.sin(state[3])) / self.length,
							state[3] + self.delta * action[1]	])

		while (state_[2] > self.theta[1]):
			state_[2] = self.theta[0] + (state_[2] - self.theta[1])

		while (state_[2] < self.theta[0]):
			state_[2] = self.theta[1] + (state_[2] - self.theta[0])

		while (state_[3] > self.steer_angle[1]):
			state_[3] = self.steer_angle[0] + (state_[3] - self.steer_angle[1])

		while (state_[3] < self.steer_angle[0]):
			state_[3] = self.steer_angle[1] + (state_[3] - self.steer_angle[0])

		return state_




env = env_car_4d()
env.algorithm_init()
# state = np.array([1, 2, math.pi / 2, math.pi / 4], dtype = float)
# action = np.array([1, math.pi / 8], dtype = float)
# print(env.state_transition(state, action))
env.value_iteration()
