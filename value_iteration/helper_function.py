import numpy as np
import math
import pandas as pd
import os
from scipy.interpolate import RegularGridInterpolator

class value_interpolation_function_car(object):
	def __init__(self):
		self.x = (-5, 5)
		self.y = (-5, 5)
		self.theta = (-math.pi, math.pi)
		self.ranges = np.array([self.x, self.y, self.theta])
		self.state_step_num = np.array([25, 25, 9])
		# self.value_file_path = os.environ['PROJ_HOME_3'] + "/value_iteration/value_epsilon.npy"
		# self.value_file_path = os.environ['PROJ_HOME_3'] + "/value_iteration/value_boltzmann.npy"
		self.value_file_path = os.environ['PROJ_HOME_3'] + "/value_iteration/value_boltzmann_angle.npy"
		print("using value file:", self.value_file_path)
		self.fill_value = -400
		self.value = None
		self.interpolating_function = None

	def setup(self):
		state_dim_num = len(self.state_step_num)
		l = []
		for size in self.state_step_num:
			x = np.empty((size), dtype = float)
			l.append(x)
		self.state_grid = np.array(l, dtype = object)

		for i in range(state_dim_num):
			self.state_grid[i] = np.linspace(self.ranges[i][0], self.ranges[i][1], self.state_step_num[i])

		self.value = np.load(self.value_file_path)
		self.interpolating_function = RegularGridInterpolator((self.state_grid[0],
																self.state_grid[1],
																self.state_grid[2]),
																self.value,
																bounds_error = False,
																fill_value = self.fill_value)

	def interpolate_value(self, v):
		return self.interpolating_function(v).astype(np.float32)


class value_interpolation_function_quad(object):
	def __init__(self):
		self.x = (-5, 5)
		self.vx = (-4, 5)
		self.z = (0, 10)
		self.vz = (-2, 5)
		self.theta = (-math.pi / 2, math.pi / 2)
		self.w = (-math.pi*5/3, math.pi*5/3)

		# self.ranges = np.array([self.x, self.z, self.vx, self.vz, self.theta, self.w])
		# self.state_step_num = np.array([11, 11, 9, 9, 11, 11])

		self.ranges = np.array([self.x,  self.vx, self.z, self.vz, self.theta, self.w])
		self.state_step_num = np.array([11, 9, 11, 9, 11, 11])

		# self.value_file_path = os.environ['PROJ_HOME_3'] + "/value_iteration/value_matrix_quad_6D/value_matrix_7.npy"

		self.value_file_path = os.environ['PROJ_HOME_3'] + "/value_iteration/value_iteration_6d_xubo_version_1/value_matrix_quad_6D/transfered_value_matrix_7.npy"
		# print("using value file:", self.value_file_path)
		self.fill_value = -400
		self.value = None
		self.interpolating_function = None

	def setup(self):
		state_dim_num = len(self.state_step_num)
		l = []
		for size in self.state_step_num:
			x = np.empty((size), dtype = float)
			l.append(x)
		self.state_grid = np.array(l, dtype = object)

		for i in range(state_dim_num):
			self.state_grid[i] = np.linspace(self.ranges[i][0], self.ranges[i][1], self.state_step_num[i])

		self.value = np.load(self.value_file_path)
		self.interpolating_function = RegularGridInterpolator((self.state_grid[0],
																self.state_grid[1],
																self.state_grid[2],
																self.state_grid[3],
																self.state_grid[4],
																self.state_grid[5]),
																self.value,
																bounds_error = False,
																fill_value = self.fill_value)

	def interpolate_value(self, v):
		return self.interpolating_function(v).astype(np.float32)


# test = value_interpolation_function_quad()
#
# test.setup()
#
# print(test.interpolate_value(np.zeros([10,6])))