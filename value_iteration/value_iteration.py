import math
import numpy as np
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm



class world_env(object):
    def __init__(self):
        ###################### World Env ####################
        self.x = (-5, 5)
        self.y = (-5, 5)

        self.gravity = 9.8

        self.obstacles = None

        ###################### Drone ########################
        self.theta = (0, 2*math.pi)
        self.omega = (-5*math.pi, 5*math.pi)

        self.vx = (-2, 2)
        self.vy = (-2, 2)

        # thrust - action range
        self.t1 = (0, 36.7875/2)
        self.t2 = (0, 36.7875/2)

        self.mass = 1.25
        self.length = 0.5
        self.inertia = 0.125
        self.trans = 0.25  # Cdv
        self.rot =  0.02255  # Cd_phi
        self.delta = 0.4

        self.dim_x = [0, 2, 4, 5]
        self.dim_y = [1, 3, 4, 5]
        self.dim_a = [6, 7]

        self.discount = 0.90
        self.threshold = 0.5

        ##################### Goal Env ######################        
        self.goal_x = (4, 5)
        self.goal_y = (4, 5)

        self.goal_vx = self.vx
        self.goal_vy = self.vy

        self.goal_theta = self.theta
        self.goal_omega = self.omega

        ############# Discreteness Coefficient ##############
        # 6D state + 2D action
        # (x, y, vx, vy, theta, omega, t1, t2)
        self.ranges = np.array([self.x, self.y, self.vx, self.vy, self.theta, self.omega, self.t1, self.t2])

        # self.step_number = np.array([10, 10, 10, 10, 10, 10, 10, 10])
        # self.step_number = np.array([2, 2, 2, 2, 2, 2, 2, 2]) # used for debug
        # self.step_number = np.array([31, 11, 11, 11, 11, 11, 15, 15]) # used for debug
        self.step_number = np.array([11, 11, 11, 11, 9, 9, 15, 15]) # used for debug
        # self.step_number = np.array([4, 4, 4, 4, 4, 4, 4, 4]) # used for debug
        # self.step_number = np.array([6, 6, 6, 6, 6, 6, 6, 6]) # used for debug 
        # self.step_number = np.array([8, 8, 8, 8, 8, 8, 8, 8]) # used for debug
        # self.step_number = np.array([12, 12, 12, 12, 12, 12, 12 ,12]) # used for debug
        # self.step_number = np.array([2, 3, 4, 5, 6, 5, 4, 3]) # used for debug


        # The list of goal dimension
        self.goals = np.array([self.goal_x, self.goal_y, self.goal_vx, self.goal_vy, self.goal_theta, self.goal_omega])

        #Grid is used for storing the n-D arrays regard to the discret states after cutting
        self.grid = None

        ############# The discrete states in the world ##############
        self.state = None
        self.value = None

        self.state_type = None
        self.state_reward = None

        # reward = [regular state, in goal, crashed, overspeed]
        self.reward = np.array([-1, 1000, -500, -200])

    def add_obstacle(self, x1, x2, y1, y2):
        if ((x1 > x2) or (y1 > y2)):
            print("Add obstacle error! Check parameters")
            return

        if (self.obstacles is None):
            self.obstacles = np.array([[x1, x2, y1, y2]], dtype=float)
        else:
            self.obstacles = np.concatenate((self.obstacles, np.array([[x1, x2, y1, y2]])), axis = 0)

    def state_cutting(self, dim = 8):
        # Initialize and declare array space
        l = []
        for size in self.step_number:
            x = np.empty((size), dtype = float)
            l.append(x)

        self.grid = np.array(l, dtype = object)


        for i in range(dim):
            if (self.ranges[i][0] < self.ranges[i][1]):
                self.grid[i] = np.linspace(self.ranges[i][0], self.ranges[i][1], self.step_number[i])


            # Range exception!
            else:
                print("State %d range error!" %(i))
                self.grid = None
                break

        if (self.obstacles is not None):
            for i in range(self.obstacles.shape[0]):
                self.obstacles[i] = self.seek_nearest_position(self.obstacles[i], dim = [0, 0, 1, 1])

        # print(self.obstacles)
        # print(self.grid)

    def state_init(self):
        self.value_x = np.zeros(self.step_number[self.dim_x])
        self.value_y = np.zeros(self.step_number[self.dim_y])

        self.value_x.fill(-2000)
        self.value_y.fill(-2000)

        self.state_type_x = np.zeros(self.step_number[self.dim_x], dtype = int)
        self.state_type_y = np.zeros(self.step_number[self.dim_y], dtype = int)

    def value_iteration(self, debug = False, pretrain_file = "", iteration_number = 0):
        # Generate the combination of 4-d data
        # Select the rows which are not in obstacles

        states = np.array(np.meshgrid(self.grid[0],
                                        self.grid[2],
                                        self.grid[4],
                                        self.grid[5])).T.reshape(-1, len(self.dim_x))


        index = []
        for i, s in enumerate(states):
            t = self.check_crash(s, self.dim_x)
            if (t != 2):
                index.append(i)

        print("The total 4D states: ", states.shape)
        states = states[index]
        print("The safe 4D states: ", states.shape)

        # Update the value array iteratively
        actions = np.array(np.meshgrid(self.grid[6],
                                        self.grid[7])).T.reshape(-1, len(self.dim_a))

        iteration = 0

        dir_path = os.path.dirname(os.path.realpath(__file__))

        if (pretrain_file != ""):
            try:
                self.value_x = np.load(pretrain_file + str(iteration_number) + ".npy")
                iteration = iteration_number + 1
                print("Load pre_trained value matrix successfully!")
            except:
                print("Load pre_trained value matrix failed")

        
        while True:

            if (debug):
                file_name = "/log/log_x_" + str(iteration) + ".txt"
                f = open(dir_path + file_name, "w")

            num_remain = 0
            num_crash = 0
            num_transition = 0 

            delta = 0

            for s in states:
                best_value = -1000000
                for a in actions:
                    s_ = self.state_transition_x(s, a)

                    if (debug):
                        log = "s: " + np.array2string(s, precision = 4, separator = '  ') + "\n"
                        f.write(log)
                        log = "s_: " + np.array2string(s_, precision = 4, separator = '  ') + "\n"
                        f.write(log)
                        log = "a: " + np.array2string(a, precision = 4, separator = '  ') + "\n"
                        f.write(log)


                    if (self.check_crash(s_, self.dim_x) == 2):
                        reward = self.reward[2]
                        s_ = s
                        num_crash += 1
                    else:
                        s_ = self.seek_nearest_position(s_, self.dim_x)
                        reward = self.reward[self.state_check(s_, self.dim_x)]


                    index = self.state_to_index(s, self.dim_x)
                    index_ = self.state_to_index(s_, self.dim_x)

                    if (index == index_):
                        num_remain += 1

                    # print(index, index_)

                    if (debug):
                        log = ''.join(str(e) + ' ' for e in index) + '\n'
                        f.write(log)
                        log = ''.join(str(e) + ' ' for e in index_) + '\n'
                        f.write(log)
                        f.write(str(self.value_x[index_[0], index_[1], index_[2], index_[3]]))
                        f.write(str(reward))
                        f.write("_________________________\n")

                    best_value = max(best_value, reward + self.discount * self.value_x[index_[0], index_[1], index_[2], index_[3]])

                    num_transition += 1
                    if (num_transition % 100000 == 0):
                        print(num_transition)

                index = self.state_to_index(s, self.dim_x)
                delta = max(delta, abs(best_value - self.value_x[index[0], index[1], index[2], index[3]]))
                self.value_x[index[0], index[1], index[2], index[3]] = best_value


            print("iteraion %d:" %(iteration))
            print("delta: ", delta)
            print("num_transition: ", num_transition)
            print("num_remain: ", num_remain - num_crash)
            print("num_crash: ", num_crash)
            print("\n\n")

            self.value_output(iteration, "state")

            np.save("./value_matrix/value_matrix" + str(iteration), self.value_x)

            iteration += 1

            if (debug):
                f.close()

            if (delta < self.threshold):
                break 

    def value_iteration_y(self, debug = False, pretrain_file = "", iteration_number = 0):
        # Generate the combination of 4-d data (y, vy, theta, omega)
        # Select the rows which are not in obstacles

        states = np.array(np.meshgrid(self.grid[1],
                                        self.grid[3],
                                        self.grid[4],
                                        self.grid[5])).T.reshape(-1, len(self.dim_y))

        index = []
        for i, s in enumerate(states):
            t = self.check_crash(s, self.dim_y)
            if (t != 2):
                index.append(i)

        print("The total 4D states: ", states.shape)
        states = states[index]
        print("The safe 4D states: ", states.shape)

        actions = np.array(np.meshgrid(self.grid[6],
                                        self.grid[7])).T.reshape(-1, len(self.dim_a))

        iteration = 0

        dir_path = os.path.dirname(os.path.realpath(__file__))

        if (pretrain_file != ""):
            try:
                self.value_y = np.load(pretrain_file + str(iteration_number) + ".npy")
                iteration = iteration_number + 1
                print("Load pre_trained value matrix successfully!")
            except:
                print("Load pre_trained value matrix failed")


        while True:
            if (debug):
                file_name = "/log/log_y_" + str(iteration) + ".txt"
                f = open(dir_path + file_name, "w")


            num_remain = 0
            num_crash = 0
            num_transition = 0
            delta = 0

            delete_list = []

            for i, s in enumerate(states):
                best_value = -1000000
                

                for a in actions:
                    s_ = self.state_transition_y(s, a)

                    if (debug):
                        log = "s: " + np.array2string(s, precision = 4, separator = '  ') + "\n"
                        f.write(log)
                        log = "s_: " + np.array2string(s_, precision = 4, separator = '  ') + "\n"
                        f.write(log)
                        log = "a: " + np.array2string(a, precision = 4, separator = '  ') + "\n"
                        f.write(log)

                    if (self.check_crash(s_, self.dim_y) == 2):
                        reward = self.reward[2]
                        s_ = s
                        num_crash += 1
                    else:
                        s_ = self.seek_nearest_position(s_, self.dim_y)
                        reward = self.reward[self.state_check(s_, self.dim_y)]

                    index = self.state_to_index(s, self.dim_y)
                    index_ = self.state_to_index(s_, self.dim_y)

                    if (index == index_):
                        num_remain += 1

                    if (debug):
                        log = ''.join(str(e) + ' ' for e in index) + '\n'
                        f.write(log)
                        log = ''.join(str(e) + ' ' for e in index_) + '\n'
                        f.write(log)
                        f.write(str(self.value_y[index_[0], index_[1], index_[2], index_[3]]))
                        f.write(str(reward))
                        f.write("__________________________\n")

                    best_value = max(best_value, reward + self.discount * self.value_y[index_[0], index_[1], index_[2], index_[3]])

                    num_transition += 1
                    # if (num_transition % 100000 == 0):
                    #     print(num_transition) 

                index = self.state_to_index(s, self.dim_y)
                current_delta = abs(best_value - self.value_y[index[0], index[1], index[2], index[3]])
                delta = max(delta, current_delta)
                if (current_delta < self.threshold):
                    # print(current_delta)
                    delete_list.append(i)

                self.value_y[index[0], index[1], index[2], index[3]] = best_value


            print("iteration %d:" %(iteration))
            print("delta: ", delta)
            print("num_transition: ", num_transition)
            print("num_remain: ", num_remain - num_crash)
            print("num_crash: ", num_crash)
            print("\n\n")

            self.value_output_y(iteration, "state")

            np.save("./value_matrix/value_matrix_y_" + str(iteration), self.value_y)

            iteration += 1

            print(states.shape)
            print(len(delete_list))
            states = np.delete(states, delete_list, 0)
            print(states.shape)

            if (debug):
                f.close()

            if (delta < self.threshold):
                break

    def value_output(self, iteration, mode = "index"):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_name = "./value_iteration/value_iteration_" + str(iteration) + ".txt"


        f = open(dir_path + file_name, "w")
        if (mode == "index"):
            for idx1 in range(self.step_number[self.dim_x[0]]):
                for idx2 in range(self.step_number[self.dim_x[1]]):
                    for idx3 in range(self.step_number[self.dim_x[2]]):
                        for idx4 in range(self.step_number[self.dim_x[3]]):
                            s = str(idx1)+ '  ' + str(idx2) + '  ' + str(idx3) + '  ' + str(idx4) + '  ' + str(self.value_x[idx1, idx2, idx3, idx4]) + '\n'
                            f.write(s)
        else:
            for idx1 in range(self.step_number[self.dim_x[0]]):
                for idx2 in range(self.step_number[self.dim_x[1]]):
                    for idx3 in range(self.step_number[self.dim_x[2]]):
                        for idx4 in range(self.step_number[self.dim_x[3]]):
                            state = self.index_to_state([idx1, idx2, idx3, idx4], self.dim_x)
                            s = np.array2string(state, precision = 4, separator = '  ')
                            s += '  ' + format(self.value_x[idx1, idx2, idx3, idx4], '.4f') + '\n'
                            f.write(s)

        f.close()

    def value_output_y(self, iteration, mode = "index"):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_name = "./value_iteration/value_iteration_y_" + str(iteration) + ".txt"

        f = open(dir_path + file_name, "w")
        if (mode == "index"):
            for idx1 in range(self.step_number[self.dim_y[0]]):
                for idx2 in range(self.step_number[self.dim_y[1]]):
                    for idx3 in range(self.step_number[self.dim_y[2]]):
                        for idx4 in range(self.step_number[self.dim_y[3]]):
                            s = str(idx1) + '  ' + str(idx2) + '  ' + str(idx3) + '  ' + str(idx4) + '  ' + str(self.value_y[idx1, idx2, idx3, idx4]) + '\n'
                            f.write(s)

        else:
            for idx1 in range(self.step_number[self.dim_y[0]]):
                for idx2 in range(self.step_number[self.dim_y[1]]):
                    for idx3 in range(self.step_number[self.dim_y[2]]):
                        for idx4 in range(self.step_number[self.dim_y[3]]):
                            state = self.index_to_state([idx1, idx2, idx3, idx4], self.dim_y)
                            s = np.array2string(state, precision = 4, separator = '  ')
                            s += '  ' + format(self.value_y[idx1, idx2, idx3, idx4], '.4f') + '\n'
                            f.write(s)

        f.close()

    def index_to_state(self, index, dim):
        state = np.zeros(len(dim), dtype = float)
        for i in range(len(dim)):
            state[i] = self.grid[dim[i]][index[i]]

        return state

    def state_to_index(self, state, dim):
        grid = self.grid[dim]

        for i in range(len(dim)):
            grid[i] = np.absolute(grid[i] - state[i])

        index = []
        for i in range(len(dim)):
            index.append(grid[i].argmin())

        return index

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
            if (d == 4):
                continue

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

    def seek_nearest_position(self, state, dim):
        # This function can find the nearest position on discrete world for one state
        # The type of state is a row of numpy array, e.g. state = np.array([2, 3, 4, 5]) a 4D state
        # The dim stores the corresponding dimension of state. 
        for i in range(len(dim)):
            state[i] = self.grid[dim[i]] [np.argmin(np.absolute(self.grid[dim[i]] - state[i]))]

        return state

    def state_transition_x(self, state, action):
        # state = [x, vx, theta, omega]

        act = np.sum(action)
        act_diff = action[0] - action[1]

        state_ = np.array([state[0] + state[1] * self.delta,
                            state[1] + self.delta / self.mass * (-self.trans * state[1] - math.sin(state[2]) * act),
                            state[2] + state[3] * self.delta,
                            state[3] + self.delta / self.inertia * (-self.rot * state[3] - self.length * act_diff)])

        while (state_[2] > self.theta[1]):
            state_[2] = self.theta[0] + (state_[2] - self.theta[1])

        while (state_[2] < self.theta[0]):
            state_[2] = self.theta[1] + (state_[2] - self.theta[0])

        return state_

    def state_transition_y(self, state, action):
        # state = [y, vy, theta, omega]

        act = np.sum(action)
        act_diff = action[0] - action[1]

        state_ = np.array([state[0] + state[1] * self.delta,
                            state[1] + self.delta * ((-self.trans * state[1] + math.cos(state[2]) * act) / self.mass - self.gravity),
                            state[2] + state[3] * self.delta,
                            state[3] + self.delta / self.inertia * (-self.rot * state[3] - self.length * act_diff)
                            ])

        while (state_[2] > self.theta[1]):
            state_[2] = self.theta[0] + (state_[2] - self.theta[1])

        while (state_[2] < self.theta[0]):
            state_[2] = self.theta[1] + (state_[2] - self.theta[0])

        return state_
                            
    def plot_result(self, dir_path, file_path):
        file = dir_path + file_path
        data = np.load(file)
        omega_index = 0

        while omega_index < data.shape[-1]:
            x = np.zeros(data.shape[:-1])
            vx = np.zeros(data.shape[:-1])
            theta = np.zeros(data.shape[:-1])
            value = np.zeros(data.shape[:-1])

            for i, d in np.ndenumerate(data):
                if (i[-1] == omega_index):
                    x[i[:-1]] = i[0]
                    vx[i[:-1]] = i[1]
                    theta[i[:-1]] = i[2]
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

            ax.set_xlabel('x', fontsize = 10)
            ax.set_ylabel('vx', fontsize = 10)
            ax.set_zlabel('theta', fontsize = 10)

            plt.show()
            fig.savefig("Omega_" + str(omega_index), dpi=fig.dpi)

            omega_index += 1

    def debug(self):
        x = -2000.0
        reward = 1000
        iteration_count = 0
        delta = 1000000
        threshold = 0.00005

        while (delta >= threshold):
            delta = abs((reward + x * 0.9) - x)
            x = reward + x * 0.9
            iteration_count += 1
            print(x)

        print("count:  ", iteration_count)  


if __name__ == "__main__":
    env = world_env()
    # env.plot_result("./value_matrix/", "value_matrix_y_117.npy")
    env.state_cutting()
    env.state_init()

    env.value_iteration_y(False, "./value_matrix/value_matrix_y_", 10)

    # env.add_obstacle(-4.5,4.5,-4.5,4.5)
    # env.add_obstacle(3,4,3,4)

    # self.vx = (-2, 2)
    # self.theta = (0, 2*math.pi)
    # self.omega = (-5*math.pi, 5*math.pi)

    # (regular, goal, crashed, overspeed)
    # [x, vx, theta, omega]
    # states = np.array([[2, 0, 1, 0],
    #                     [-6, 0, 1, 0],
    #                     [4.5, -3, 1, 0],
    #                     [4.5, 0, 1, 0],
    #                     [4.5, 0, 0, 0],
    #                     [4.4, 1, 1.3, 2.14],
    #                     [3.5, 1, 1.3, 2],
    #                     [0, -3, 2, -30],
    #                     [0, 0, 0, -30]
    #                     ], dtype = float)

    # for state in states:
    #     print(env.state_check(state, env.dim_x))

    # env.value_iteration(False, "./value_matrix/value_matrix", 95)
    # env.value_iteration(False)

    # env.value_output("state")

    # state = np.array([5, 0.8, 0.7854, -3.1416], dtype = float)
    # action = np.array([17.0799, 14.4522], dtype = float)
    # print(env.state_transition(state, action))

    # env.state_to_index([1,1,1,1], env.dim_x)
    # print(env.grid[env.dim_x])
    # print(env.state_to_index(np.array([0.3, 0.8, 1, 0.1]), env.dim_x))

    # state = np.array([2, 3, 4, 5])
    # action = np.array([1, 1])
    # print(env.state_transition(state, action))


    # TO DO LIST:
    # Optimize the performance of algorithm in implementation
    
    

