import math
import numpy as np
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import RegularGridInterpolator
import time

class env_quadrotor_3d(object):
	def __init__(self):
		########### World Setting ########## 
		self.x = (-5, 5)
		self.z = (0, 10)
		self.gravity = 9.81
		self.obstacles = None

		########### Drone Setting ##########
		self.vx = (-5, 5)
		self.vz = (-6, 10)
		self.theta = (-math.pi, math.pi)
		self.omega = (-2*math.pi, 2*math.pi)
		self.ranges = np.array([self.x, self.z, self.theta])

		self.mass = 1.27
		self.length = 0.4
		self.delta = 0.2

		self.actions = np.array([self.vx, self.vz, self.omega])
		self.all = np.array([self.x, self.z, self.theta, self.vx, self.vz, self.omega])
		########### Goal Setting ##########
		self.goal_x = (3, 5)
		self.goal_z = (8, 10)
		self.goal_theta = self.theta
		self.goals = np.array([self.goal_x, self.goal_z, self.goal_theta])

		########### Algorithm Setting ##########
		self.discount = 0.90
		self.threshold = 0.5

		# reward = [regula state, in goal, crashed, overspeed]
		self.reward_list = np.array([0, 1000, -400, -200], dtype = float)

		########## Discreteness Setting ##########
		# 3D state
		# 3D action
		# (x, z, theta)
		# (vx, vz, omega)
		self.state_step_num = np.array([21, 31, 9])
		self.action_step_num = np.array([5, 5, 6])
		self.all_step_num = np.concatenate((self.state_step_num, self.action_step_num), axis = None)

		########## Algorithm Declaration ##########
		self.state_grid = None
		self.action_grid = None

		self.value = None
		self.qvalue = None
		self.reward = None

		self.state_type = None
		self.state_reward = None

		self.acts = None

	def add_obstacle(self, x1, x2, z1, z2):
		if ((x1 > x2)  or  (z1 > z2)):
			print("Add obstacle error!")
			return

		if (self.obstacles is None):
			self.obstacles = np.array([[x1, x2, z1, z2]], dtype = float)
		else:
			self.obstacles = np.concatenate((self.obstacles, np.array([[x1, x2, z1, z2]])), axis = 0)

	def algorithm_init(self):
		self.add_obstacles()
		self.state_discretization()
		self.state_init()
		self.action_init()
 		
	def add_obstacles(self):
		obstacles = [[-2.75, -1.25, 4.25, 5.75],
					[-0.75, 0.75, 8.0, 9.0],
					[2.0, 5.0, 4.25, 5.75]]

		for obs in obstacles:
			self.add_obstacle(obs[0], obs[1], obs[2], obs[3])

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
											self.state_grid[2])).T.reshape(-1, len(self.state_grid))

		self.value = np.zeros(self.state_step_num, dtype = float)
		self.qvalue = np.zeros(self.all_step_num, dtype = float)
		self.reward = np.zeros(self.states.shape[0], dtype = float)

		delete_list = []		

		for i, s in enumerate(self.states):
			state_type = self.state_check(s)
			if (state_type == 1  or  state_type == 2):
				index = tuple(self.state_to_index(s))
				self.value[index] = self.reward_list[state_type]
				self.reward[i] = self.reward_list[state_type]
				self.qvalue[index[0], index[1], index[2], :, :, :] = self.reward_list[state_type]
				# print("i: ", i ,"value: ", self.qvalue[17, 9, 0, 0, 0, 0], "s: ", s, "index: ", index)
				# if (index[0] == 17  and  index[1] == 9):
				# 	print(s, state_type, index, self.reward_list[state_type])
				# 	print(self.qvalue[17, 9, 0, :, :, :])
				# # delete_list.append(i)

		# for i, s in enumerate(self.reward):
		# 	if (self.reward[i] == 1000):
		# 		print(self.states[i], self.reward[i])

		# print(self.qvalue[17, 9, 0, :, :, :])

	def action_init(self):
		self.acts = np.array(np.meshgrid(self.action_grid[0],
										self.action_grid[1],
										self.action_grid[2])).T.reshape(-1, self.actions.shape[0])

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

	def check_crash(self, s):
		# return 2 = crashed
		# return 0 = no crash

		if (self.obstacles is not None and len(self.obstacles)):
			for obs in self.obstacles:
				if (s[0] >= obs[0]  and  s[0]  <= obs[1]  and  
					s[1] >= obs[2]  and  s[1]  <= obs[3]):
					return 2

		if (s[0] < self.ranges[0][0]  or  s[0] > self.ranges[0][1]):
			return 2
		if (s[1] < self.ranges[1][0]  or  s[1] > self.ranges[1][1]):
			return 2

		return 0

	def check_crash_with_boundary(self, s, width):
		# return 2 = crashed
		# return 0 = no crash

		if (self.obstacles is not None and len(self.obstacles)):
			for obs in self.obstacles:
				if (s[0] >= obs[0] - width  and  s[0]  <= obs[1] + width  and  
					s[1] >= obs[2] - width  and  s[1]  <= obs[3] + width):
					return 2

		if (s[0] < self.ranges[0][0] + width  or  s[0] > self.ranges[0][1] - width):
			return 2
		if (s[1] < self.ranges[1][0] + width  or  s[1] > self.ranges[1][1] - width):
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
																			sub_states[2]),
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

	def value_output(self, iteration, readable_file = False):
		dir_path = "./value_matrix_quadrotor_3D/"
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

		########## Output Q-value Matrix ##########
		file_name = "q_value_matrix" + "_" + str(iteration)
		np.save(dir_path + file_name, self.qvalue)
		if (readable_file == True):
			file_name = file_name + ".txt"
			f = open(dir_path + file_name, "w")

			for _, s in enumerate(self.states):
				for _, a in enumerate(self.acts):
					index_s = tuple(self.state_to_index(s))
					index_a = tuple(self.action_to_index(a))
					index_all = self.combine_to_tuple(index_s, index_a)
					st = np.array2string(np.concatenate((s, a), axis = None), precision = 4, separator = '  ')
					st += '  ' + format(self.qvalue[index_all], '.4f') + '\n'
					f.write(st)

			f.close()


	def combine_to_tuple(self, s, a):
		arr = []
		for v in s:
			arr.append(v)
		for v in a:
			arr.append(v)
		return tuple(arr)

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

		return self.value[r[0][0]:r[0][1], r[1][0]:r[1][1], r[2][0]:r[2][1]], sub_states

	def state_transition(self, state, action):
		state_ = np.array([state[0] + action[0] * self.delta,
							state[1] + action[1] * self.delta,
							state[2] + action[2] * self.delta])

		while (state_[2] > self.theta[1]):
			state_[2] = self.theta[0] + (state_[2] - self.theta[1])

		while (state_[2] < self.theta[0]):
			state_[2] = self.theta[1] + (state_[2] - self.theta[0])

		return state_

	def plot_3D_result(self, dir_path, file_name, mode = "direct"):
		file = dir_path + file_name
		data = np.load(file)

		save_path = "./plots_quadrotor_3D/plot_3D"
		try:
			os.makedirs(save_path)
		except:
			print(save_path + " exists!")

		if (mode == "max"):
			x = np.zeros(data.shape[:-1])
			y = np.zeros(data.shape[:-1])
			value = np.full(data.shape[:-1], 0)

			for i, d in np.ndenumerate(data):
				x[i[:-1]] = self.state_grid[0][i[0]]
				y[i[:-1]] = self.state_grid[1][i[1]]
				value[i[:-1]] = max(value[i[:-1]], d)

			fig = plt.figure()
			ax = fig.gca(projection='3d')
			surf = ax.plot_surface(x, y, value, cmap=cm.coolwarm,
					linewidth=0, antialiased=False)
			fig.colorbar(surf, shrink=0.5, aspect=5)

			ax.set_xlabel('x', fontsize = 15)
			ax.set_ylabel('y', fontsize = 15)
			ax.set_zlabel('value', fontsize = 15)

			plt.show()

		if (mode == "average"):
			x = np.zeros(data.shape[:-1])
			y = np.zeros(data.shape[:-1])
			value = np.full(data.shape[:-1], 0)

			for i, d in np.ndenumerate(data):
				x[i[:-1]] = self.state_grid[0][i[0]]
				y[i[:-1]] = self.state_grid[1][i[1]]
				value[i[:-1]] += d

			value = value / 9

			fig = plt.figure()
			ax = fig.gca(projection='3d')
			surf = ax.plot_surface(x, y, value, cmap=cm.coolwarm,
					linewidth=0, antialiased=False)
			fig.colorbar(surf, shrink=0.5, aspect=5)

			ax.set_xlabel('x', fontsize = 15)
			ax.set_ylabel('y', fontsize = 15)
			ax.set_zlabel('value', fontsize = 15)

			plt.show()

	def generate_samples_interpolate(self, n):
		data = []
		crash = 0

		while (len(data) < n):
			sample = []
			for dim in self.all:
				v = np.random.uniform(dim[0], dim[1], 1)
				sample.append(v)
			sample = np.array(sample, dtype = float).reshape(-1)
			if (self.check_crash_with_boundary(sample[:3], width = 0.1) == 2):
				crash += 1
				continue

			data.append(sample)

		print(crash)
		data = np.array(data, dtype = float)

		dir_path = "./value_matrix_quadrotor_3D/"
		file_name = "q_value_matrix_3.npy"
		self.qvalue = np.load(dir_path + file_name)

		interpolating_function = RegularGridInterpolator((self.state_grid[0],
															self.state_grid[1],
															self.state_grid[2],
															self.action_grid[0],
															self.action_grid[1],
															self.action_grid[2]),
															self.qvalue,
															bounds_error = False,
															fill_value = self.reward_list[2])

		qvalue = np.empty((n), dtype = float)
		for i, d in enumerate(data):
			qvalue[i] = interpolating_function(d)

		dataset = pd.DataFrame({'x': data[:, 0],
								'vx': data[:, 3],
								'z': data[:, 1],
								'vz': data[:, 4],
								'phi': data[:, 2],
								'w': data[:, 5],
								'value': qvalue})
	
		dataset.to_csv("./qvalue.csv")

	def fill_mpc_table(self):
		file_path = "./test_samps_150_right_cleaned.csv"
		# file_path = "./test_data.csv"
		value_file_name = "./value_matrix_quadrotor_3D/q_value_matrix_3.npy"

		data = pd.read_csv(file_path)
		self.qvalue = np.load(value_file_name)

		# print(max(data.vx))
		# print(min(data.vx))

		interpolating_function = RegularGridInterpolator((self.state_grid[0],
															self.state_grid[1],
															self.state_grid[2],
															self.action_grid[0],
															self.action_grid[1],
															self.action_grid[2]),
															self.qvalue,
															bounds_error = False,
															fill_value = self.reward_list[2])

		qvalue = np.empty(data.shape[0], dtype = float)
		for index, row in data.iterrows():
			x = np.array([row.x, row.z, row.phi, row.vx, row.vz, row.w])
			value = interpolating_function(x)
			qvalue[index] = value
		data['value'] = qvalue
		# data.to_csv("./mpc_filled_value.csv")
		data.to_csv("./test_samps_150_right_cleaned_filled.csv")




env = env_quadrotor_3d()
env.algorithm_init()
# env.generate_samples_interpolate(20000)
# env.value_output(0, True)
# env.value_iteration()
# env.plot_3D_result("./value_matrix_quadrotor_3D", "/value_matrix_3.npy", "max")
env.fill_mpc_table()


