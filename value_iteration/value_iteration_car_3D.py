import math
import numpy as np
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import RegularGridInterpolator
import time

class env_dubin_car_3d(object):
	def __init__(self):
		########### World Setting ########## 
		self.x = (-3, 5)
		self.y = (-5, 5)
		self.obstacles = None
		self.cylinders = None

		########### Drone Setting ##########
		self.theta = (-math.pi, math.pi)
		self.ranges = np.array([self.x, self.y, self.theta])

		self.v = 1
		self.delta = 0.1

		self.omega = (-2, 2)
		self.actions = np.array([self.omega])
		########### Goal Setting ##########
		self.goal_x = (-3, -1.05)
		self.goal_y = (3.4, 4.6)
		self.goal_theta = self.theta
		self.goals = np.array([self.goal_x, self.goal_y, self.goal_theta])

		########### Algorithm Setting ##########
		self.discount = 0.995
		self.threshold = 0.5

		# reward = [regular state, in goal, crashed, overspeed]
		self.reward_list = np.array([0, 2000, -1000, -200], dtype = float)

		########## Discreteness Setting ##########
		# 3D state
		# 1D action
		# (x, y, theta)
		# (omega)
		self.state_step_num = np.array([61, 81, 11])
		# self.state_step_num = np.array([11, 11, 11])
		# self.state_step_num = np.array([31, 31, 11])
		self.action_step_num = np.array([11])
		self.dim = np.array([0, 1, 2])

		########## Algorithm Declaration ##########
		self.state_grid = None
		self.action_grid = None

		self.value = None
		self.reward = None
		self.flag = None

		self.state_type = None
		self.state_reward = None

	def add_obstacle(self, x1, x2, y1, y2):
		if ((x1 > x2) or (y1 > y2)):
			print("Add obstacle error! Check parameters")
			return

		if (self.obstacles is None):
			self.obstacles = np.array([[x1, x2, y1, y2]], dtype = float)
		else:
			self.obstacles = np.concatenate((self.obstacles, np.array([[x1, x2, y1, y2]])), axis = 0)

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

	def state_init(self):
		self.states = np.array(np.meshgrid(self.state_grid[0],
											self.state_grid[1],
											self.state_grid[2])).T.reshape(-1, len(self.dim))

		self.value = np.zeros(self.state_step_num[self.dim], dtype = float)
		self.reward = np.zeros(self.states.shape[0], dtype = float)
		self.flag = np.ones(self.states.shape[0], dtype = bool)

		# print(np.sum(self.flag))

		for i, s in enumerate(self.states):
			state_type = self.state_check(s, self.dim)
			if (state_type == 1):
				index = tuple(self.state_to_index(s, self.dim))
				self.value[index] = self.reward_list[state_type]
				self.flag[i] = False


		grid_value = np.zeros(self.state_step_num[self.dim[:2]], dtype = float)


		danger = []
		counter = 0
		t = time.time()
		for i, x in enumerate(self.state_grid[0]):
			for j, y in enumerate(self.state_grid[1]):
				for _, c in enumerate(self.cylinders):
					if (self.distance([x, y], c[:2]) <= c[2]):
						grid_value[i, j] = self.reward_list[2]
						counter += 1
						danger.append([x,y])
						break

		# x1, x2, y1, y2
		grid_value = self.add_board(grid_value, -2.99, -1.98, 3, 3.37674)

		# print("counter : ", counter)
		# print(len(danger))

		danger = np.array(danger, dtype = float)
		scatter = plt.scatter(danger[:, 0], danger[:, 1], c="red", s=3)
		plt.show()

		# print(np.sum(self.flag))

		for i, s in enumerate(self.states):
			if (self.flag[i] == False):
				continue

			index = tuple(self.state_to_index(s, self.dim))
			if (grid_value[index[:2]] < 0):
				# print(index)
				self.value[index] = grid_value[index[:2]]
				self.flag[i] = False

		# print(np.sum(self.flag))
		print(time.time() - t)

		print("The number of states: ", self.states.shape)
		print("The number of untrained states: ", np.sum(self.flag))

	def add_board(self, grid_value, x1, x2, y1, y2):
		for i, x in enumerate(self.state_grid[0]):
			for j, y in enumerate(self.state_grid[1]):
				if (x1 <= x  and  x <= x2  and  y1 <= y  and  y <= y2):
					grid_value[i, j] = self.reward_list[2]
					# print(x, y, x1, x2, y1, y2)	

		return grid_value

	def distance(self, point, cylinder):
		return math.sqrt( (point[0] - cylinder[0]) ** 2 + (point[1] - cylinder[1]) ** 2)

	def action_init(self):
		self.acts = self.action_grid[0]

	def value_iteration(self):
		self.algorithm_init()

		iteration = 0
		while True:
			delta = 0
			for i, s in enumerate(self.states):
				if (self.flag[i] == False):
					continue

				best_value = -1000000
				state_type = self.state_check(s, self.dim)
				if (state_type > 0):
					continue

				current_reward = self.reward[i]

				for i, a in enumerate(self.acts):
					s_ = self.state_transition(s, a)

					next_step_type = self.state_check(s_, self.dim)

					if (next_step_type >= 2):
						next_step_value = (next_step_type == 2) * self.reward_list[2] + (next_step_type == 3) * self.reward_list[3]
					else:
						sub_value_matrix, sub_states = self.seek_neighbors_values(s_, self.dim)
						interpolating_function = RegularGridInterpolator((sub_states[0],
																			sub_states[1],
																			sub_states[2]),
																			sub_value_matrix,
																			bounds_error = False,
																			fill_value = self.reward_list[2])
						next_step_value = interpolating_function(s_)

					best_value = max(best_value, current_reward + self.discount * next_step_value)

				index = tuple(self.state_to_index(s, self.dim))
				current_delta = abs(best_value - self.value[index])
				delta = max(delta, current_delta)
				self.value[index] = best_value


			self.value_output(iteration, True)
			print("iteraion %d:" %(iteration))
			print("delta: ", delta)
			print("\n\n")			

			iteration += 1

			if (delta < self.threshold):
				break

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

		return self.value[r[0][0]:r[0][1], r[1][0]:r[1][1], r[2][0]:r[2][1]], sub_states

	def value_output(self, iteration_number, readable_file = False):
		dir_path = "./value_matrix_car_3D/"
		try:
			os.mkdir(dir_path)
		except:
	   		print(dir_path + " exist!")

		file_name = "value_matrix" + "_" + str(iteration_number)
		print(file_name)

		np.save(dir_path + file_name, self.value)

		if (readable_file == True):
			file_name = file_name + ".txt"
			f = open(dir_path + file_name, "w")

			for i, s in enumerate(self.states):
				index = tuple(self.state_to_index(s, self.dim))
				s = np.array2string(s, precision = 4, separator = '  ')
				s += '  ' + format(self.value[index], '.4f') + '\n'
				f.write(s)

			f.close()

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

		return temp

	def check_goal(self, s, dim):
		for i, d in enumerate(dim):
			if (s[i] < self.goals[d][0] or s[i] > self.goals[d][1]):
				return 0

		return 1

	def check_crash(self, s, dim):
		# return 2 = crashed
		# return 0 = no crash

		if (self.obstacles is not None and len(self.obstacles)):
			for obs in self.obstacles:
				if (s[0] >= obs[0] and s[0] <= obs[1]):
					return 2
				if (s[1] >= obs[2] and s[1] <= obs[3]):
					return 2

		if (s[0] < self.ranges[0][0] or s[0] > self.ranges[0][1]):
			return 2
		if (s[1] < self.ranges[1][0] or s[1] > self.ranges[1][1]):
			return 2

		return 0

	def state_to_index(self, state, dim):
		grid = self.state_grid[dim]

		for i in range(len(dim)):
			grid[i] = np.absolute(grid[i] - state[i])

		index = []
		for i in range(len(dim)):
			index.append(grid[i].argmin())

		return index

	def state_transition(self, state, action):
		state_ = np.array([state[0] + self.delta * self.v * math.cos(state[2]),
							state[1] + self.delta * self.v * math.sin(state[2]),
							state[2] + self.delta * action])

		while (state_[2] > self.theta[1]):
			state_[2] = self.theta[0] + (state_[2] - self.theta[1])

		while (state_[2] < self.theta[0]):
			state_[2] = self.theta[1] + (state_[2] - self.theta[0])

		return state_

	def algorithm_init(self, system = 0):
		self.state_discretization()
		self.state_init()
		self.action_init()

	def plot_2D_result(self, dir_path, file_name):
		save_path = "./plots_cars_3D/plot_2D/"
		data = np.load(dir_path + file_name)
		values = np.sort(data.reshape(-1))
		plt.plot(values)

		try:
			os.makedirs(save_path)
		except:
			print(save_path + " exists!")

		plt.savefig(save_path + file_name.split(".")[0])

	def plot_3D_result(self, dir_path, file_name, system = 0, mode = "direct"):
		file = dir_path + file_name
		data = np.load(file)

		save_path = "./plots_cars_3D/plot_3D/"
		try:
			os.makedirs(save_path)
		except:
			print(save_path + " exist!")


		if (mode == "direct"):
			theta_index = 0
			while theta_index < data.shape[-1]:
				omega_index = 0

				x = np.zeros(data.shape[:-1])
				y = np.zeros(data.shape[:-1])
				value = np.zeros(data.shape[:-1])

				for i, d in np.ndenumerate(data):
					if (i[-1] == theta_index):
						x[i[:-1]] = self.state_grid[0][i[0]]
						y[i[:-1]] = self.state_grid[1][i[1]]
						value[i[:-1]] = d


				fig = plt.figure()
				ax = fig.gca(projection='3d')

				surf = ax.plot_surface(x, y, value, cmap=cm.coolwarm,
					   linewidth=0, antialiased=False)
				fig.colorbar(surf, shrink=0.5, aspect=5)

				ax.set_xlabel('x', fontsize = 15)
				ax.set_ylabel('y', fontsize = 15)
				ax.set_zlabel('value', fontsize = 15)
				title = "theta: " + str(round(self.state_grid[2][theta_index], 2))
				ax.set_title(title, fontsize = 15)

				plt.show()
				fig.savefig(save_path + "theta_" + str(theta_index), dpi=fig.dpi)

				theta_index += 1

		if (mode == "average"):
			x = np.zeros(data.shape[:-1])
			y = np.zeros(data.shape[:-1])
			value = np.full(data.shape[:-1], 0)

			for i, d in np.ndenumerate(data):
				x[i[:-1]] = self.state_grid[0][i[0]]
				y[i[:-1]] = self.state_grid[1][i[1]]
				value[i[:-1]] += d

			value = value / 11

			fig = plt.figure()
			ax = fig.gca(projection='3d')
			surf = ax.plot_surface(x, y, value, cmap=cm.coolwarm,
					linewidth=0, antialiased=False)
			fig.colorbar(surf, shrink=0.5, aspect=5)

			ax.set_xlabel('x', fontsize = 15)
			ax.set_ylabel('y', fontsize = 15)
			ax.set_zlabel('value', fontsize = 15)

			plt.show()

	def plot_4D_result(self, dir_path, file_name, system = 0):
		file = dir_path + file_name
		data = np.load(file)

		save_path = "./plots_cars_3D/plot_4D"
		try:
			os.makedirs(save_path)
		except:
			print(save_path + " exist!")

		omega_index = 0
		# 4d plots
		x = np.zeros(data.shape)
		y = np.zeros(data.shape)
		theta = np.zeros(data.shape)
		value = np.zeros(data.shape)

		for i, d in np.ndenumerate(data):
			x[i] = self.state_grid[0][i[0]]
			y[i] = self.state_grid[1][i[1]]
			theta[i] = self.state_grid[2][i[2]]
			value[i] = d

		x = x.reshape(-1)
		y = y.reshape(-1)
		theta = theta.reshape(-1)
		value = value.reshape(-1)

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		img = ax.scatter(x, y, theta, c=value, cmap=plt.hot())
		fig.colorbar(img)

		# ax.view_init(45,60)

		ax.set_xlabel('x', fontsize = 15)
		ax.set_ylabel('y', fontsize = 15)
		ax.set_zlabel('theta', fontsize = 15)

		plt.show()
		fig.savefig(save_path + "/4D_plot", dpi=fig.dpi)

	def add_circle_obstacles(self):
		dir_path = "./"
		file_name = "track2_scaled.csv"

		cylinders = pd.read_csv(dir_path + file_name)
		cylinders_inner = cylinders[['inner_x', 'inner_y']]
		cylinders_outer = cylinders[['outer_x', 'outer_y']]

		cylinders_inner = cylinders_inner.to_numpy()
		cylinders_outer = cylinders_outer.to_numpy()

		self.cylinders = np.zeros((cylinders_inner.shape[0] + cylinders_outer.shape[0], 3), dtype = float)
		self.cylinders[:, :-1] = np.concatenate( (cylinders_inner, cylinders_outer), axis = 0)
		self.cylinders[:, -1] = 0.1

		plt.scatter(self.cylinders[:, 0], self.cylinders[:, 1], s=1, marker='o', c='green')

		# Goal region
		plt.scatter(cylinders_inner[-1, 0], cylinders_inner[-1, 1], s=25, marker='o', c='blue')
		plt.scatter(cylinders_inner[640, 0], cylinders_inner[640, 1], s=25, marker='o', c='blue')
		plt.scatter(cylinders_outer[640, 0], cylinders_outer[640, 1], s=25, marker='o', c='blue')
		plt.scatter(cylinders_outer[-1, 0], cylinders_outer[-1, 1], s=25, marker='o', c='blue')

		# Start region
		plt.scatter(cylinders_inner[1, 0], cylinders_inner[1, 1], s=25, marker='o', c='black')
		plt.scatter(cylinders_outer[16, 0], cylinders_outer[16, 1], s=25, marker='o', c='black')
		plt.scatter(cylinders_outer[1, 0], cylinders_outer[1, 1], s=25, marker='o', c='black')
		plt.scatter(cylinders_inner[16, 0], cylinders_inner[16, 1], s=25, marker='o', c='black')



		# plt.show()



if __name__  ==  "__main__":
	env = env_dubin_car_3d()
	env.add_circle_obstacles()
	env.value_iteration()
	# env.algorithm_init()
	# env.plot_2D_result("./value_matrix_car_3D/", "value_matrix_30.npy")
	# env.plot_3D_result("./value_matrix_car_3D/", "value_matrix_30.npy")
	# env.plot_4D_result("./value_matrix_car_3D/", "value_matrix_30.npy")
