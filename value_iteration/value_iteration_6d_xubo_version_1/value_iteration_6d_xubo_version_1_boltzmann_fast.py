import math
import numpy as np
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import RegularGridInterpolator
import time
import scipy.ndimage as ndimage

# import collection

class env_quad_6d(object):
	def __init__(self):
		########### Quadrotor Setting ##########
		self.x = (-5, 5)
		self.z = (0, 10)

		self.vx = (-4, 5)
		self.vz = (-2, 5)

		# self.vx = (-7, 7)
		# self.vz = (-7, 7)

		self.theta = (-math.pi / 2, math.pi / 2)
		self.w = (-math.pi*5/3, math.pi*5/3)
		# self.w = (-math.pi * 2, math.pi * 2)

		self.t1 = (7, 12)
		self.t2 = (7, 12)

		self.ranges = np.array([self.x, self.z, self.vx, self.vz, self.theta, self.w])
		self.actions = np.array([self.t1, self.t2])
		self.dim = np.array([0, 1, 2, 3, 4, 5])


		self.length = 0.3
		self.mass = 1.25
		self.inertia = 0.125
		self.trans = 0.25 # Cdv
		self.rot = 0.02255 # Cd_phi
		########### Goal Setting ##########
		self.goal_x = (3, 5)
		self.goal_z = (8, 10)
		# self.goal_theta = (-np.pi / 4, 0)
		# self.goal_theta = (-np.pi / 3, np.pi / 3)
		self.goal_theta = (-np.pi / 6, np.pi / 3)

		self.goals = np.array([self.goal_x, self.goal_z, self.goal_theta])

		########### Algorithm Setting ##########
		self.discount = 0.998
		self.threshold = 2
		self.delta = 0.3
		self.tau = 1
		self.gravity = 9.8

		# reward = [regula state, in goal, crashed, turnover]
		self.turnover = (-1.4, 1.4)
		self.reward_list = np.array([0, 1000, -400, -800], dtype = float)

		########## Discreteness Setting ##########
		# 4D state
		# 2D action
		# (x, y, theta, steer_angle)
		# (v, omega)
		self.state_step_num = np.array([11, 11, 9, 9, 11, 11])
		# self.state_step_num = np.array([5, 5, 5, 5, 11, 11]) # for test
		self.action_step_num = np.array([6, 6])
		self.all_step_num = np.concatenate((self.state_step_num, self.action_step_num), axis = None)

		########## Algorithm Declaration ##########
		self.state_grid = None
		self.action_grid = None

		self.value = None
		self.update = None
		self.order = None

		self.state_type = None
		self.state_reward = None

		self.acts = None

		self.obstacles = None


	def algorithm_init(self):
		self.add_obstacles()
		self.state_discretization()
		self.state_init()
		self.action_init()

	def add_obstacles(self):
		# self.obstacles = np.array([[-2.75, -1.25, 4.25, 5.75], # [x0, x1, y0, y1]
		# 							[0.25, 1.75, 8.0, 9.0], 
		# 							[2.25, 5.25, 4.25, 5.75], 
		# 							[-0.25, 0.25, 0, 2]], dtype = float)

		self.obstacles = np.array([[-2.75, -1.25, 4.25, 5.75],		# [x0, x1, y0, y1]
									[-0.75, 0.75, 8.0, 9.0],		# only this one changed
									[2.25, 5.25, 4.25, 5.75],
									[-0.25, 0.25, 0, 2]], dtype=float)
		print(self.obstacles)

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

		print(self.state_grid)


	def state_init(self):
		self.states = np.array(np.meshgrid(self.state_grid[0],
											self.state_grid[1],
											self.state_grid[2],
											self.state_grid[3],
											self.state_grid[4],
											self.state_grid[5])).T.reshape(-1, len(self.state_grid))

		self.value = np.zeros(self.state_step_num, dtype = float)
		self.update = np.zeros(self.states.shape[0], dtype = bool)
		self.order =  np.zeros(self.states.shape, dtype = float)

		ordered = np.zeros(self.states.shape[0], dtype = bool)

		#### Calculate the distance between states and goal ####
		# x = 3
		# y = 8
		# radius = 0.5
		# total_count = 0
		# while (total_count < self.states.shape[0]):
		# 	for i, s in enumerate(self.states):
		# 		if (ordered[i]):
		# 			continue
		# 		d = (s[0] - x) * (s[0] - x) + (s[1] - y) * (s[1] - y)
		# 		if (math.sqrt(d) <= radius):
		# 			self.order[total_count] = self.states[i]
		# 			total_count += 1
		# 			ordered[i] = True
		# 	radius += 0.5
		# 	print("current radius: ", radius, "current ordered: ", total_count)

		# self.states = self.order
		########################################################

		counter_0 = 0
		counter_1 = 0
		counter_2 = 0
		counter_3 = 0
 
		for i, s in enumerate(self.states):
			state_type = self.state_check(s, self.dim)
			if (state_type > 0):
				index = tuple(self.state_to_index(s))
				self.value[index] = self.reward_list[state_type]
			
			self.update[i] = (state_type == 0)
			counter_0 += int(state_type == 0)
			counter_1 += int(state_type == 1)
			counter_2 += int(state_type == 2)
			counter_3 += int(state_type == 3)

		print(len(self.states))
		print("update states: ", counter_0)
		print("goal states: ", counter_1)
		print("crashed states: ", counter_2)
		print("turnover states: ", counter_3)




	def action_init(self):
		self.acts = np.array(np.meshgrid(self.action_grid[0],
										self.action_grid[1])).T.reshape(-1, self.actions.shape[0])
		

	def state_check(self, s, dim):
		temp = 0
		temp = self.check_goal(s)
		if (temp):
			return temp

		temp = self.check_crash(s)
		if (temp):
			return temp

		temp = self.check_turnover(s[4])
		if (temp):
			return temp

		return temp

	def check_goal(self, s):
		for i, v in enumerate(s[:2]):
			if (v < self.goals[i][0]  or  v > self.goals[i][1]):
				return 0

		if (s[4] < self.goals[2][0]  or  s[4] > self.goals[2][1]):
			return 0 

		return 1

	def check_crash(self, s):
		# return 2 = crashed
		# return 0 = no crash

		# width = 0.2
		# width = 0.25
		width = 0.225
		# print("using width:", width)
		
		if (self.obstacles is not None  and  len(self.obstacles)):
			for obs in self.obstacles:
				if (s[0] >= obs[0] - width  and  
					s[0] <= obs[1] + width  and
					s[1] >= obs[2] - width  and  
					s[1] <= obs[3] + width):
					return 2

		if (s[0] <= self.ranges[0][0] + width  or  
			s[0] >= self.ranges[0][1] - width):
			return 2

		if (s[1] < self.ranges[1][0] + width  or  
			s[1] > self.ranges[1][1] - width):
			return 2

		return 0 

	def check_turnover(self, s):
		# return 3 = turnover
		# return 0 = in normal

		if (s < self.turnover[0]  or  s > self.turnover[1]):
			return 3

		return 0

	def check_cross(self, s, s_):
		k = (s_[1] - s[1]) / (s_[0] - s[0])

		if (np.absolute(k) > 10000 or np.absolute(k) < 0.0001):
			return False

		b = s[1] - k*s[0]

		
		if (self.obstacles is not None  and  len(self.obstacles)):
			for obs in self.obstacles:

				if ( (s[0] >= obs[0]  and  s[0] <= obs[1])  or  (s_[0] >= obs[0]  and  s_[0] <= obs[1]) ):
					y = k * obs[0] + b
					if ( y >= obs[2]  and  y <= obs[3]):
						return True
			
					y = k * obs[1] + b
					if ( y >= obs[2]  and  y <= obs[3]):
						return True

				if ( (s[1] >= obs[2]  and  s[1] <= obs[3])  or  (s_[1] >= obs[2]  and  s_[1] <= obs[3]) ):			
					x = (obs[2] - b) / k
					if ( x >= obs[0]  and  x <= obs[1]):
						return True

				
					x = (obs[3] - b) / k
					if ( x >= obs[0]  and  x <= obs[1]):
						return True

		return False


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

	def state_transition(self, s, a):
		action_sum = np.sum(a)
		action_diff = a[0] - a[1]

		state_ = np.array([s[0] + s[2] * self.delta,
							s[1] + s[3] * self.delta,
							s[2] + self.delta * (-self.trans * s[2] + math.sin(s[4]) * action_sum) / self.mass,
							s[3] + self.delta * (-self.gravity + (-self.trans * s[3] + math.cos(s[4]) * action_sum) / self.mass),
							s[4] + s[5] * self.delta,
							s[5] + self.delta * (-self.rot * s[5] + self.length * action_diff) / self.inertia])

		while (state_[4] > math.pi):
			state_[4] = -math.pi + (state_[4] - math.pi)

		while (state_[4] < -math.pi):
			state_[4] = math.pi + (state_[4] + math.pi)

		return state_



	def value_iteration(self, mode = "default"):
		iteration = 0

		while True:
			delta = 0
			# interpolating_function = RegularGridInterpolator((self.state_grid[0],
			# 													self.state_grid[1],
			# 													self.state_grid[2],
			# 													self.state_grid[3],
			# 													self.state_grid[4],
			# 													self.state_grid[5]),
			# 													self.value,
			# 													bounds_error = False,
			# 													fill_value = self.reward_list[2])

			# total_time = 0
			# states_counter = 0
			# interpolation_time = 0

			for i, s in enumerate(self.states):
				if (self.update[i] == False):
					continue

				if (mode == "boltzmann"):
					best_value = 0
					qvalue = np.array([], dtype = float)
					bolt = np.array([], dtype = float)
				else:
					best_value = -10000

				for _, a in enumerate(self.acts):
					s_ = self.state_transition(s, a)
					next_step_type = self.state_check(s_, self.dim)
					#### Obstacle crossing judgement ####

					if (next_step_type <= 1):
						if (self.check_cross(s, s_)):
							next_step_type = 2

					if (next_step_type >= 2):
						next_step_value = self.reward_list[next_step_type]
					else:
						sub_value_matrix, sub_states = self.seek_neighbors_values(s_, self.dim)

						#### Old version interpolation function ####

						# interpolating_function = RegularGridInterpolator((sub_states[0],
						# 													sub_states[1],
						# 													sub_states[2],
						# 													sub_states[3],
						# 													sub_states[4],
						# 													sub_states[5]),
						# 													sub_value_matrix,
						# 													bounds_error = False,
						# 													fill_value = self.reward_list[2])
						# next_step_value = interpolating_function(s_)

						####################################

						#### ndimgae map coordinates ####
						sub_states = np.array(sub_states, dtype = float)
						s_ = (s_ - sub_states[:, 0]) / (sub_states[:, 1] - sub_states[:, 0])
						s_ = np.array([s_],dtype=float)

						next_step_value = ndimage.map_coordinates(sub_value_matrix, s_.T, mode = "constant", cval = self.reward_list[2])

						# interpolation_time += time.time() - temp_t
						##################################


					if (mode == "boltzmann"):
						temp_value = self.discount * next_step_value
						qvalue = np.append(qvalue, [[temp_value]])
						# 100 is a scalar to solve math range error. The value range becomes [-4, 10]
						bolt = np.append(bolt, [[math.exp(temp_value / 100 / self.tau)]])
					else:
						best_value = max(best_value, self.discount * next_step_value)

				index = tuple(self.state_to_index(s))
				if (mode == "boltzmann"):
					total = bolt.sum()
					p_a = bolt / total
					best_value = np.matmul(p_a, np.transpose(qvalue))

				current_delta = abs(best_value - self.value[index])
				delta = max(delta, current_delta)

				self.value[index] = best_value

				# states_counter += 1
				# total_time += time.time() - t
				# if (states_counter % 10000 == 0):
				# 	print(total_time)
				# 	print(interpolation_time)

			self.value_output(iteration, True)
			print("iteraion %d:" %(iteration))
			print("delta: ", delta)
			print("\n\n")			

			iteration += 1

			if (delta < self.threshold):
				break

	def seek_neighbors_values(self, state, dim):
		index = self.state_to_index(state)
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

		return self.value[r[0][0]:r[0][1], r[1][0]:r[1][1], r[2][0]:r[2][1], r[3][0]:r[3][1], r[4][0]:r[4][1], r[5][0]:r[5][1]], sub_states


	def value_output(self, iteration_number, readable_file = False):
		# dir_path = "./value_matrix_quad_6D/"
		dir_path = "/local-scratch/xlv/SL_optCtrl/value_iteration/value_iteration_6d_xubo_version_1/value_matrix_quad_6D_boltzmann_fast_airspace_201910_ddpg/trial_3/"
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
				index = tuple(self.state_to_index(s))
				s = np.array2string(s, precision = 4, separator = '  ')
				s += '  ' + format(self.value[index], '.4f') + '\n'
				f.write(s)

			f.close()

	def plot_2D_result(self, mode = "max"):
		# file = "./value_matrix_quad_6D/value_matrix_7.npy"
		# file = "./value_matrix_10.npy"
		file = "/local-scratch/xlv/SL_optCtrl/value_iteration/value_iteration_6d_xubo_version_1/value_matrix_quad_6D_boltzmann_fast_airspace_201910_ddpg/trial_3/value_matrix_9.npy"
		data = np.load(file)

		save_path = "./"
		try:
			os.makedirs(save_path)
		except:
			print(save_path + " exist!")

		print(data.shape)

		if (mode == "direct"):
			theta_index = 0


			while theta_index < data.shape[2]:
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

				theta_index += 1
				# plt.show()
				fig.savefig(save_path + "hard_task_boltzmann_" + "theta_" + str(theta_index), dpi=fig.dpi)


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


		if (mode == "max"):
			x = np.zeros(data.shape[:2])
			y = np.zeros(data.shape[:2])
			value = np.full(data.shape[:2], -800)

			for i, d in np.ndenumerate(data):
				x[i[:2]] = self.state_grid[0][i[0]]
				y[i[:2]] = self.state_grid[1][i[1]]
				value[i[:2]] = max(value[i[:2]], d)

			fig = plt.figure()
			ax = fig.gca(projection='3d')
			surf = ax.plot_surface(x, y, value, cmap=cm.coolwarm,
					linewidth=0, antialiased=False)
			fig.colorbar(surf, shrink=0.5, aspect=5)

			ax.set_xlabel('x', fontsize = 15)
			ax.set_ylabel('y', fontsize = 15)
			ax.set_zlabel('value', fontsize = 15)

			plt.show()

	def test_interpolation(self):
		self.value = np.load("./value_matrix_quad_6D/value_matrix_10.npy")

		### Sample data ###

		n = 10000
		data = []
		while (len(data) < n):
			sample = []
			for d in range(len(self.ranges)):
				v = np.random.uniform(self.ranges[d][0], self.ranges[d][1], 1)
				sample.append(v)
			sample = np.array(sample, dtype = float).reshape(-1)
			if (self.check_crash(sample) == 2):
				continue
			data.append(sample)

		data = np.array(data, dtype = float)		


		### Old Interpolation version ###

		t = time.time()
		
		final_0 = []
		for s in data:
			sub_value_matrix, sub_states = self.seek_neighbors_values(s, self.dim)
			test_if = RegularGridInterpolator((sub_states[0],
												sub_states[1],
												sub_states[2],
												sub_states[3],
												sub_states[4],
												sub_states[5]),
												sub_value_matrix,
												bounds_error = False,
												fill_value = self.reward_list[2])
			value = test_if(s)
			final_0.append(value)
			# print(value)


		print("scipy Regular grid interpolation: ", time.time() - t)


		### New Interpolation version ###
		import scipy.ndimage as ndimage
		t = time.time()

		final_1 = []
		for s in data:
			sub_value_matrix, sub_states = self.seek_neighbors_values(s, self.dim)
			sub_states = np.array(sub_states, dtype = float)
			s = (s - sub_states[:, 0]) / (sub_states[:, 1] - sub_states[:, 0])
			s = np.array([s],dtype=float)

			value = ndimage.map_coordinates(sub_value_matrix, s.T, mode = "constant", cval = -400)
			# print(value)
			final_1.append(value)

		print("ndimage map coordinates method: ", time.time() - t)

		print(np.array([final_0, final_1], dtype = float).T)

		# data = np.array([[0, 1], [2, 3]], dtype = float)
		# import scipy.ndimage as ndimage
		# v = ndimage.map_coordinates(data, np.array([[0.5, 0.5]]).T)
		# print(v)
		

	def generate_samples_interpolate(self, n):
		data = []
		while (len(data) < n):
			sample = []
			for d in range(len(self.ranges)):
				v = np.random.uniform(self.ranges[d][0], self.ranges[d][1], 1)
				sample.append(v)
			sample = np.array(sample, dtype = float).reshape(-1)
			if (self.check_crash(sample) == 2):
				continue
			data.append(sample)

		data = np.array(data, dtype = float)

		file_name = "./value_matrix_quad_6D/value_matrix_12.npy"

		self.value = np.load(file_name)
		interploating_function = RegularGridInterpolator((self.state_grid[0],
															self.state_grid[1],
															self.state_grid[2],
															self.state_grid[3],
															self.state_grid[4],
															self.state_grid[5]),
															self.value,
															bounds_error = False,
															fill_value = self.reward_list[2])

		value = np.empty((n), dtype = float)
		for i, d in enumerate(data):
			value[i] = interploating_function(d)

		dataset = pd.DataFrame({'x': data[:, 0],
								'vx': data[:, 2],
								'z': data[:, 1],
								'vz': data[:, 3],
								'theta': data[:, 4],
								'w': data[:, 5],
								'value': value})
		dataset = dataset[['x', 'vx', 'z', 'vz', 'theta', 'w', 'value']]
		dataset.to_csv("./quad_6D_value_iteration_samples.csv")

	def refill_values(self):
		state_file = "./valFunc_vi_filled_cleaned.csv"
		value_file = "./value_matrix_quad_6D/value_matrix_12.npy"

		self.value = np.load(value_file)
		df = pd.read_csv(state_file)

		for index, row in df.iterrows():
			s = row[['x', 'z', 'vx', 'vz', 'phi', 'w']]
			s = s.to_numpy()

			sub_value_matrix, sub_states = self.seek_neighbors_values(s, self.dim)
			sub_states = np.array(sub_states, dtype = float)
			s = (s - sub_states[:, 0]) / (sub_states[:, 1] - sub_states[:, 0])
			s = np.array([s],dtype=float)

			value = ndimage.map_coordinates(sub_value_matrix, s.T, mode = "constant", cval = self.reward_list[2])
			df = df.set_value(index, 'value', value)

		df.to_csv("./new_version_interpolation_valFunc_vi_filled_cleaned.csv")



env = env_quad_6d()
env.algorithm_init()
# env.generate_samples_interpolate(100000)
# env.value_iteration(mode="boltzmann")
# env.refill_values()
env.plot_2D_result(mode="max")
# s = np.array([0.3, 8.05, -4, 5, 0, -5.236], dtype = float)
# s_ = np.array([0.2, 7.95, -4, 5, 0, -5.236], dtype = float)
# print(env.check_cross(s, s_))
# a = np.array([12, 12], dtype =float)
# print(env.state_transition(s, a))
# env.test_interpolation()