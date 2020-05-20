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
        self.x = (-5, 5)
        self.y = (-5, 5)
        self.obstacles = None
        self.cylinders = None

        ########### Drone Setting ##########
        self.theta = (-math.pi, math.pi)
        self.ranges = np.array([self.x, self.y, self.theta])

        self.v = (-2, 2)
        self.delta = 0.6

        self.omega = (-1, 1)
        self.actions = np.array([self.v, self.omega])
        ########### Goal Setting ##########
        self.goal_center = (3.5, 3.5)
        self.goal_radius = 1.0
        self.goal_theta = (math.pi / 2, math.pi * 3 / 4)

        ########### Algorithm Setting ##########
        self.discount = 0.93
        self.negative_discount = 0.5
        self.epsilon = 0.3
        self.threshold = 0.5
        self.tau = 1

        # reward = [regular state, in goal, crashed, overspeed]
        self.reward_list = np.array([0, 1000, -400, -400], dtype=float)
        self.beta = np.array([1, 1.8, 0.005, 0])

        ########## Discreteness Setting ##########
        # 3D state
        # 1D action
        # (x, y, theta)
        # (omega)
        # self.state_step_num = np.array([5, 5, 10])
        self.state_step_num = np.array([25, 25, 9])
        # self.state_step_num = np.array([31, 31, 11])
        self.action_step_num = np.array([9, 9])
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
            self.obstacles = np.array([[x1, x2, y1, y2]], dtype=float)
        else:
            self.obstacles = np.concatenate((self.obstacles, np.array([[x1, x2, y1, y2]])), axis=0)

    def state_discretization(self):
        state_dim_num = len(self.state_step_num)
        action_dim_num = len(self.action_step_num)

        ########### Space Declaration ##########
        l = []
        for size in self.state_step_num:
            x = np.empty((size), dtype=float)
            l.append(x)
        self.state_grid = np.array(l, dtype=object)

        l = []
        for size in self.action_step_num:
            x = np.empty((size), dtype=float)
            l.append(x)
        self.action_grid = np.array(l, dtype=object)

        ########## Discretization Procedure ##########
        for i in range(state_dim_num):
            self.state_grid[i] = np.linspace(self.ranges[i][0], self.ranges[i][1], self.state_step_num[i])
        for i in range(action_dim_num):
            self.action_grid[i] = np.linspace(self.actions[i][0], self.actions[i][1], self.action_step_num[i])

    # print(self.state_grid)
    # print(self.action_grid)

    def state_init(self):
        self.states = np.array(np.meshgrid(self.state_grid[0],
                                           self.state_grid[1],
                                           self.state_grid[2])).T.reshape(-1, len(self.dim))

        self.value = np.zeros(self.state_step_num[self.dim], dtype=float)
        self.reward = np.zeros(self.states.shape[0], dtype=float)

        for i, s in enumerate(self.states):
            state_type = self.state_check(s, self.dim)
            if (state_type >= 1):
                index = tuple(self.state_to_index(s, self.dim))
                self.value[index] = self.reward_list[state_type]

    def action_init(self):
        self.acts = np.array(np.meshgrid(self.action_grid[0],
                                         self.action_grid[1])).T.reshape(-1, 2)

    def algorithm_init(self):
        self.add_obstacle(-0.35, 0.35, -0.35, 0.35)
        self.add_obstacle(2.65, 3.35, -1.35, -0.65)
        self.add_obstacle(1.65, 2.35, 0.65, 1.35)
        self.add_obstacle(-2.35, -1.65, -2.35, -1.65)
        self.add_obstacle(-0.35, 0.35, 3.65, 4.35)
        self.add_obstacle(-3.35, -2.65, 1.65, 2.35)

        self.state_discretization()
        self.state_init()
        self.action_init()

    def value_iteration(self, mode="positive"):
        iteration = 0
        while True:
            delta = 0
            for i, s in enumerate(self.states):
                if (mode == "positive"):
                    best_value = -1000000
                if (mode == "negative"):
                    best_value = 1000000
                if (mode == "epsilon"):
                    best_value = -1000000
                    qvalue = np.array([], dtype=float)
                if (mode == "average" or mode == "boltzmann"):
                    best_value = 0
                    qvalue = np.array([], dtype=float)
                if (mode == "boltzmann"):
                    bolt = np.array([], dtype=float)

                state_type = self.state_check(s, self.dim)
                if (state_type > 0):
                    continue

                current_reward = self.reward[i]

                for i, a in enumerate(self.acts):
                    s_ = self.state_transition(s, a)
                    next_step_type = self.state_check(s_, self.dim)
                    if (next_step_type >= 2):
                        next_step_value = self.reward_list[next_step_type]
                    else:
                        sub_value_matrix, sub_states = self.seek_neighbors_values(s_, self.dim)
                        interpolating_function = RegularGridInterpolator((sub_states[0],
                                                                          sub_states[1],
                                                                          sub_states[2]),
                                                                         sub_value_matrix,
                                                                         bounds_error=False,
                                                                         fill_value=self.reward_list[2])
                        next_step_value = interpolating_function(s_)
                    if (mode == "positive"):
                        best_value = max(best_value, current_reward + self.discount * next_step_value)

                    if (mode == "epsilon"):
                        v = current_reward + self.discount * next_step_value
                        best_value = max(best_value, v)
                        qvalue = np.append(qvalue, [[v]])

                    if (mode == "boltzmann"):
                        temp_value = current_reward + self.discount * next_step_value
                        qvalue = np.append(qvalue, [[temp_value]])
                        # 100 is a scalar to solve math range error. The value range becomes [-4, 10]
                        bolt = np.append(bolt, [[math.exp(temp_value / 100 / self.tau)]])

                    if (mode == "negative"):
                        best_value = min(best_value, current_reward + self.negative_discount * next_step_value)

                    if (mode == "average"):
                        best_value = best_value + next_step_value

                index = tuple(self.state_to_index(s, self.dim))
                if (mode == "average"):
                    best_value = best_value / self.acts.shape[0]
                if (mode == "epsilon"):
                    best_value = (1 - self.epsilon) * best_value + self.epsilon * qvalue.sum() / self.acts.shape[0]
                if (mode == "boltzmann"):
                    total = bolt.sum()
                    p_a = bolt / total
                    best_value = np.matmul(p_a, np.transpose(qvalue))

                current_delta = abs(best_value - self.value[index])
                delta = max(delta, current_delta)

                self.value[index] = best_value

            self.value_output(iteration, True, mode)
            print("iteraion %d:" % (iteration))
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
            if (left == 0 or left == self.state_step_num[dim[i]] - 1):
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

    def value_output(self, iteration_number, readable_file=False, mode="positive"):
        dir_path = "./value_matrix_car_3D_2/"
        try:
            os.mkdir(dir_path)
        except:
            print(dir_path + " exist!")

        file_name = "value_matrix" + "_" + mode + "_" + str(iteration_number)
        print(file_name)

        np.save(dir_path + file_name, self.value)

        if (readable_file == True):
            file_name = file_name + ".txt"
            f = open(dir_path + file_name, "w")

            for i, s in enumerate(self.states):
                index = tuple(self.state_to_index(s, self.dim))
                s = np.array2string(s, precision=4, separator='  ')
                s += '  ' + format(self.value[index], '.4f') + '\n'
                f.write(s)

            f.close()

    def state_check(self, s, dim):
        # This function is used for returning the state_type of a state
        # Also including check whether the state is out of range

        # temp = (0, 1, 2, 3) -> (regular state, in goal, crashed, overspeed)
        temp = 0
        # Check in goal range
        temp = self.check_goal(s)
        if (temp):
            return temp

        temp = self.check_crash(s)
        if (temp):
            return temp

        return temp

    def check_goal(self, s):
        d = math.sqrt((s[0] - self.goal_center[0]) ** 2 + (s[1] - self.goal_center[1]) ** 2)
        if (d <= self.goal_radius and s[2] >= self.goal_theta[0] and s[2] <= self.goal_theta[1]):
            return 1
        return 0

    def check_crash(self, s, width=0.2):
        # return 2 = crashed
        # return 0 = no crash

        if (self.obstacles is not None and len(self.obstacles)):
            for obs in self.obstacles:
                if (s[0] >= obs[0] - width and
                        s[0] <= obs[1] + width and
                        s[1] >= obs[2] - width and
                        s[1] <= obs[3] + width):
                    return 2

        if (s[0] < self.ranges[0][0] + width or s[0] > self.ranges[0][1] - width):
            return 2
        if (s[1] < self.ranges[1][0] + width or s[1] > self.ranges[1][1] - width):
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
        state_ = np.array([state[0] + self.delta * action[0] * math.cos(state[2]),
                           state[1] + self.delta * action[0] * math.sin(state[2]),
                           state[2] + self.delta * action[1]])

        while (state_[2] > self.theta[1]):
            state_[2] = self.theta[0] + (state_[2] - self.theta[1])

        while (state_[2] < self.theta[0]):
            state_[2] = self.theta[1] + (state_[2] - self.theta[0])

        return state_

    def plot_3D_result(self, dir_path, file_name, mode="direct"):
        file = dir_path + file_name
        data = np.load(file)

        save_path = "./plots_cars_3D_2/plot_3D/"
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

                ax.set_xlabel('x', fontsize=15)
                ax.set_ylabel('y', fontsize=15)
                ax.set_zlabel('value', fontsize=15)
                title = "theta: " + str(round(self.state_grid[2][theta_index], 2))
                ax.set_title(title, fontsize=15)

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

            ax.set_xlabel('x', fontsize=15)
            ax.set_ylabel('y', fontsize=15)
            ax.set_zlabel('value', fontsize=15)

            plt.show()

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

            ax.set_xlabel('x', fontsize=15)
            ax.set_ylabel('y', fontsize=15)
            ax.set_zlabel('value', fontsize=15)

            plt.show()

    def generate_samples_interpolate(self, n):
        data = []
        while (len(data) < n):
            sample = []
            for d in range(len(self.ranges)):
                v = np.random.uniform(self.ranges[d][0], self.ranges[d][1], 1)
                sample.append(v)
            sample = np.array(sample, dtype=float).reshape(-1)
            if (self.check_crash(sample) == 2):
                continue
            data.append(sample)

        data = np.array(data, dtype=float)

        dir_path = "./value_matrix_car_3D_2/"
        file_name = "value_matrix_boltzmann_30.npy"

        self.value = np.load(dir_path + file_name)
        interploating_function = RegularGridInterpolator((self.state_grid[0],
                                                          self.state_grid[1],
                                                          self.state_grid[2]),
                                                         self.value,
                                                         bounds_error=False,
                                                         fill_value=self.reward_list[2])

        value = np.empty((n), dtype=float)
        for i, d in enumerate(data):
            value[i] = interploating_function(d)

        dataset = pd.DataFrame({'x': data[:, 0],
                                'y': data[:, 1],
                                'theta': data[:, 2],
                                'value': value})
        dataset.to_csv("./car_3D_harder_task_samples_value.csv")

    def matrix_accumulation(self):
        positive_file_path = "./value_matrix_car_3D_2/value_matrix_positive_13.npy"
        negative_file_path = "./value_matrix_car_3D_2/value_matrix_negative_4.npy"
        save_file_path = "./value_matrix_car_3D_2/final_matrix.npy"

        positive_value = np.load(positive_file_path)
        negative_value = np.load(negative_file_path)
        final_value = positive_value + negative_value
        np.save(save_file_path, final_value)

    def reset(self):
        while True:
            sample = []
            for d in range(len(self.ranges)):
                v = np.random.uniform(self.ranges[d][0], self.ranges[d][1], 1)
                sample.append(v)
            sample = np.array(sample, dtype=float).reshape(-1)
            if (self.check_crash(sample) != 2):
                break
        return sample

    def set_interpolation(self, valM):
        interploating_function = RegularGridInterpolator((self.state_grid[0],
                                                        self.state_grid[1],
                                                        self.state_grid[2]),
                                                        valM,
                                                        bounds_error=False,
                                                        fill_value=self.reward_list[2])
        return interploating_function

if __name__ == "__main__":
    env = env_dubin_car_3d()
    # env.add_obstacle(-0.35, 0.35, -0.35, 0.35)
    # env.add_obstacle(3.15, 3.85, -0.85, -0.15)
    # env.add_obstacle(1.65257, 2.35257, -1.94221, -1.24221)
    # env.add_obstacle(1.707, 2.407, 0.645, 1.345)
    # env.add_obstacle(-2.518, -1.818, -2.569, -1.869)
    # env.add_obstacle(-0.35, 0.35, 3.15, 3.85)
    # env.add_obstacle(-3.35, -2.65, 1.65, 2.35)

    env.add_obstacle(-0.35, 0.35, -0.35, 0.35)
    env.add_obstacle(2.65, 3.35, -1.35, -0.65)
    env.add_obstacle(1.65, 2.35, 0.65, 1.35)
    env.add_obstacle(-2.35, -1.65, -2.35, -1.65)
    env.add_obstacle(-0.35, 0.35, 3.65, 4.35)
    env.add_obstacle(-3.35, -2.65, 1.65, 2.35)

    # (0, 0) l = 0.7
    # (3.5, -0.5) l = 0.7
    # (2.057, 0.995)
    # (-2.168, -2.219)
    # (0, 3.5)
    # (-3, 2)

    env.algorithm_init()
    # env.generate_samples_interpolate(n = 20000)
    # env.plot_3D_result("./value_matrix_car_3D_2/", "value_matrix_negative_4.npy", "direct")
    # env.plot_3D_result("./value_matrix_car_3D_2/", "value_matrix_average_36.npy", "direct")
    # env.plot_3D_result("./value_matrix_car_3D_2/", "value_matrix_epsilon_20.npy", "direct")
    # env.plot_3D_result("./value_matrix_car_3D_2/", "value_matrix_boltzmann_30.npy", "direct")

    # env.plot_3D_result("./value_matrix_car_3D_2/", "final_matrix.npy", "direct")
    # env.value_iteration(mode = "epsilon")
    # env.value_iteration()
    # env.value_iteration(mode = "boltzmann")

    # env.value_iteration(mode = "negative")
    # env.matrix_accumulation()

    env.generate_samples_interpolate(n=50000)
