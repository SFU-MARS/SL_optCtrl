import numpy as np
import math
import pandas as pd
import os
from scipy.interpolate import RegularGridInterpolator

class value_interpolation_function(object):
	def __init__(self):
		self.x = (-5, 5)
		self.y = (-5, 5)
		self.theta = (-math.pi, math.pi)
		self.ranges = np.array([self.x, self.y, self.theta])
		self.state_step_num = np.array([25, 25, 9])
		self.value_file_path = os.environ['PROJ_HOME_3'] + "/value_iteration/value_matrix.npy"
		self.fill_value = -1000
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

# test = value_interpolation_function()
#
# test.setup()
#
# print(test.interpolate_value(np.zeros([10,3])))