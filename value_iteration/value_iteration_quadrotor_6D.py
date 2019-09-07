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
		# self.goal_theta = (0.45, 1.05)
		self.goal_theta = self.theta
		self.goal_omega = self.omega
		self.goals = np.array([self.goal_x, self.goal_z, self.goal_vx, self.vz, self.goal_theta, self.goal_omega])

		########### Algorithm Setting ##########
		self.discount = 0.90
		self.threshold = 0.5

		# reward = [regula state, in goal, crashed, overspeed]
		self.reward_list = np.array([0, 1000, -400, -200], dtype = float)

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

		self.value = None
		self.reward = None

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
		if (system == 0):
			self.states = np.array(np.meshgrid(self.state_grid[0],
												self.state_grid[2],
												self.state_grid[4],
												self.state_grid[5])).T.reshape(-1, len(self.dim[system]))
		else:
			self.states = np.array(np.meshgrid(self.state_grid[1],
												self.state_grid[3],
												self.state_grid[4],
												self.state_grid[5])).T.reshape(-1, len(self.dim[system]))


		self.value = np.zeros(self.state_step_num[self.dim[system]], dtype = float)
		self.reward = np.zeros(self.states.shape[0], dtype = float)
		delete_list = []

		# print(self.states.shape)

		for i, s in enumerate(self.states):
			state_type = self.state_check(s, self.dim[system])
			if (state_type == 1):
				index = tuple(self.state_to_index(s, self.dim[system]))
				self.value[index] = self.reward_list[state_type]
				delete_list.append(i)

		# print(self.states.shape)

		# self.states = np.delete(self.states, delete_list, 0)
		# self.reward = np.delete(self.reward, delete_list, 0)

		print("The number of states: ", self.states.shape)

	def action_init(self):
		self.acts = np.array(np.meshgrid(self.action_grid[0],
										self.action_grid[1])).T.reshape(-1, self.actions.shape[0])

	def value_iteration(self, system):
		self.algorithm_init(system)

		iteration = 0
		while True:
			delta = 0
			for i, s in enumerate(self.states):
				best_value = -1000000
				state_type = self.state_check(s, self.dim[system])
				if (state_type > 0):
					continue

				current_reward = self.reward[i]

				for i, a in enumerate(self.acts):
					if (system == 0):
						s_ = self.state_transition_x(s, a)
					else:
						s_ = self.state_transition_z(s, a)

					next_step_type = self.state_check(s_, self.dim[system])

					if (next_step_type >= 2):
						next_step_value = (next_step_type == 2) * self.reward_list[2] + (next_step_type == 3) * self.reward_list[3]
					else:
						sub_value_matrix, sub_states = self.seek_neighbors_values(s_, self.dim[system])
						interpolating_function = RegularGridInterpolator((sub_states[0],
																			sub_states[1],
																			sub_states[2],
																			sub_states[3]),
																			sub_value_matrix,
																			bounds_error = False,
																			fill_value = self.reward_list[2])
						next_step_value = interpolating_function(s_)

					best_value = max(best_value, current_reward + self.discount * next_step_value)

				index = tuple(self.state_to_index(s, self.dim[system]))
				current_delta = abs(best_value - self.value[index])
				delta = max(delta, current_delta)
				self.value[index] = best_value


			self.value_output(system, iteration, True)
			print("iteraion %d:" %(iteration))
			print("delta: ", delta)
			print("\n\n")			

			iteration += 1

			if (delta < self.threshold):
				break

	def value_output(self, system, iteration_number, readable_file = False):
		dir_path = "./value_matrix/"
		try:
			os.mkdir(dir_path)
		except:
	   		print(dir_path + " exist!")

		file_name = "value_matrix" + "_" + str(system) + "_" + str(iteration_number)
		print(file_name)

		np.save(dir_path + file_name, self.value)

		if (readable_file == True):
			file_name = file_name + ".txt"
			f = open(dir_path + file_name, "w")

			for i, s in enumerate(self.states):
				index = tuple(self.state_to_index(s, self.dim[system]))
				s = np.array2string(s, precision = 4, separator = '  ')
				s += '  ' + format(self.value[index], '.4f') + '\n'
				f.write(s)

			f.close()

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

	def state_transition_z(self, state, action):
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

	def seek_neighbors_values(self, state, dim):
		index = self.state_to_index(state, dim)
		r = []
		sub_states = []

		for i in range(len(dim)):
			left = index[i]
			right = 0
			if (left == 0  or  left == self.state_step_num[dim[i]] - 1):
				right = 1 if left == 0 else left - 1
			else:
				if (self.state_grid[dim[i]][left + 1] - state[i]) < (state[i] - self.state_grid[dim[i]][left - 1]):
					right = left + 1
				else:
					right = left - 1
			left, right = left - (left > right), right + (left > right)
			right += 1
			r.append([left, right])
			sub_states.append(self.state_grid[dim[i]][left:right])	

		if (dim[0] == 0):
			return self.value[r[0][0]:r[0][1], r[1][0]:r[1][1], r[2][0]:r[2][1], r[3][0]:r[3][1]], sub_states
		else:
			return self.value[r[0][0]:r[0][1], r[1][0]:r[1][1], r[2][0]:r[2][1], r[3][0]:r[3][1]], sub_states

	def state_check(self, s, dim):
		# This function is used for returning the state_type of a state
		# Also including check whether the state is out of range

		# temp = (0, 1, 2, 3) -> (regular state, in goal, crashed, overspeed)
		temp = 0
		# Check in goal range
		temp = self.check_goal(s, dim)
		if (temp):
			return temp

		temp = self.check_crash(s, dim)
		if (temp):
			return temp

		temp = self.check_overspeed(s, dim)
		if (temp):
			return temp

		return temp

	def check_goal(self, s, dim):
		for i, d in enumerate(dim):
			if (s[i] < self.goals[d][0] or s[i] > self.goals[d][1]):
				return 0

		return 1

	def check_crash(self, s, dim):
		# return 1 = crashed
		# return 0 = no crash

		if (self.obstacles is not None and len(self.obstacles)):
			temp_obs = None
			if (dim[0] == 0):
				temp_obs = self.obstacles[:, :2]
			else:
				temp_obs = self.obstacles[:, 2:]

			for obs in temp_obs:
				if (s[0] >= obs[0] and s[0] <= obs[1]):
					return 2

		if (s[0] < self.ranges[dim[0]][0] or s[0] > self.ranges[dim[0]][1]):
			return 2

		return 0

	def check_overspeed(self, s, dim):
		# return 3 = overspeed
		# return 0 = regular speed
		# Only check the velocity and angular velocity
		for i, d in enumerate(dim):
			if (i == 0 or i == 2): # pass the x and theta variable
				continue

			if (s[i] < self.ranges[d][0] or s[i] > self.ranges[d][1]):
				return 3

		return 0

	def state_to_index(self, state, dim):
		grid = self.state_grid[dim]

		for i in range(len(dim)):
			grid[i] = np.absolute(grid[i] - state[i])

		index = []
		for i in range(len(dim)):
			index.append(grid[i].argmin())

		return index

	def plot_2D_result(self, dir_path, file_name):
		save_path = "./plots_quadrotor_6D/plot_2D/"
		data = np.load(dir_path + file_name)
		values = np.sort(data.reshape(-1))
		plt.plot(values)

		try:
			os.makedirs(save_path)
		except:
			print(save_path + " exists!")

		plt.savefig(save_path + file_name.split(".")[0])
		plt.clf()

	def plot_3D_result(self, dir_path, file_name, system = 0):
		file = dir_path + file_name
		data = np.load(file)

		theta_index = 0
		while theta_index < data.shape[-1]:
			omega_index = 0

			save_path = "./plots_quadrotor_6D/plot_3D_" + str(system) + "/theta_" + str(theta_index)
			try:
				os.makedirs(save_path)
			except:
				print(save_path + " exist!")

			while omega_index < data.shape[-2]:
				print(theta_index, omega_index)
				x = np.zeros(data.shape[:-2])
				vx = np.zeros(data.shape[:-2])
				value = np.zeros(data.shape[:-2])

				for i, d in np.ndenumerate(data):
					if (i[-1] == omega_index  and  i[-2] == theta_index):
						x[i[:-2]] = self.state_grid[0 + system][i[0]]
						vx[i[:-2]] = self.state_grid[2 + system][i[1]]
						value[i[:-2]] = d


				fig = plt.figure()
				ax = fig.gca(projection='3d')

				surf = ax.plot_surface(x, vx, value, cmap=cm.coolwarm,
					   linewidth=0, antialiased=False)
				fig.colorbar(surf, shrink=0.5, aspect=5)

				ax.set_xlabel('x/z', fontsize = 15)
				ax.set_ylabel('vx/vz', fontsize = 15)
				ax.set_zlabel('value', fontsize = 15)
				title = "theta: " + str(round(self.state_grid[4][theta_index], 2)) + "   omega: " + str(round(self.state_grid[5][omega_index], 2))
				ax.set_title(title, fontsize = 15)

				# plt.show()
				print(save_path)
				fig.savefig(save_path + "/omega_" + str(omega_index), dpi=fig.dpi)

				omega_index += 1
			theta_index += 1

	def plot_4D_result(self, dir_path, file_name, system = 0):
		file = dir_path + file_name
		data = np.load(file)

		save_path = "./plots_quadrotor_6D/plot_4D_" + str(system)
		try:
			os.makedirs(save_path)
		except:
			print(save_path + " exist!")

		omega_index = 0
		# 4d plots
		while omega_index < data.shape[-1]:
			x = np.zeros(data.shape[:-1])
			vx = np.zeros(data.shape[:-1])
			theta = np.zeros(data.shape[:-1])
			value = np.zeros(data.shape[:-1])

			for i, d in np.ndenumerate(data):
				if (i[-1] == omega_index):
					x[i[:-1]] = self.state_grid[0 + system][i[0]]
					vx[i[:-1]] = self.state_grid[2 + system][i[1]]
					theta[i[:-1]] = self.state_grid[4][i[2]]
					value[i[:-1]] = d
	 
			x = x.reshape(-1)
			vx = vx.reshape(-1)
			theta = theta.reshape(-1)
			value = value.reshape(-1)

			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')

			img = ax.scatter(x, vx, theta, c=value, cmap=plt.hot())
			fig.colorbar(img)

			# ax.view_init(45,60)

			ax.set_xlabel('x/z', fontsize = 15)
			ax.set_ylabel('vx/vz', fontsize = 15)
			ax.set_zlabel('theta', fontsize = 15)

			# plt.show()
			fig.savefig(save_path + "/theta_" + str(omega_index), dpi=fig.dpi)

			omega_index += 1

	def fill_table(self, csv_path, dir_path, num_x = 0, num_z = 0):
		data = pd.read_csv(csv_path)

		if (dir_path):
			file_x = dir_path + "value_matrix_0_" + str(num_x) + ".npy"
			file_z = dir_path + "value_matrix_1_" + str(num_z) + ".npy"

			try:
				value_x = np.load(file_x)
				value_z = np.load(file_z)
			except:
				print("Failed to reload value matrix!")

		min_value = min(np.min(value_x), np.min(value_z))

		interpolating_function_x = RegularGridInterpolator((self.state_grid[0],
															self.state_grid[2],
															self.state_grid[4],
															self.state_grid[5]), 
															value_x, 
															bounds_error = False, 
															fill_value = min_value)

		interpolating_function_y = RegularGridInterpolator((self.state_grid[1],
															self.state_grid[3],
															self.state_grid[4],
															self.state_grid[5]), 
															value_z, 
															bounds_error = False, 
															fill_value = min_value)

		data = data.astype({'value': 'float'})


		for index, row in data.iterrows():
			state_x = np.array([row.x, row.vx, row.phi, row.w])
			state_z = np.array([row.z, row.vz, row.phi, row.w])
			value_x = interpolating_function_x(state_x)
			value_z = interpolating_function_y(state_z)
			value = min(value_x, value_z)

			data.at[index, 'value'] = value

		data.to_csv("./refactor2_valueFunc_train_linear_filled.csv")

	def algorithm_init(self, system = 0):
		env.state_discretization()
		env.state_init(system)
		env.action_init()


env = env_quadrotor_6d()
# env.value_iteration(0)
# env.value_iteration(1)

env.algorithm_init()

# env.plot_2D_result("./value_matrix/", "value_matrix_0_26.npy")
# env.plot_2D_result("./value_matrix/", "value_matrix_1_32.npy")
# env.plot_3D_result("./value_matrix/", "value_matrix_0_26.npy", 0)
# env.plot_3D_result("./value_matrix/", "value_matrix_1_32.npy", 1)
# env.plot_4D_result("./value_matrix/", "value_matrix_0_26.npy", 0)
# env.plot_4D_result("./value_matrix/", "value_matrix_1_32.npy", 1)
env.fill_table("./valueFunc_train.csv", "./value_matrix/", 26, 32)







