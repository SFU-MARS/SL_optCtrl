import math
import numpy as np
import pandas as pd
import os


class world_env(object):
    def __init__(self):
        ###################### World Env ####################
        self.x = (-5, 5)
        self.y = (-5, 5)

        self.gravity = 9.8

        self.obstacles = None

        ##################### Goal Env ######################        
        self.goal_x = (4, 5)
        self.goal_y = (4, 5)

        self.goal_vx = (-10, 10)
        self.goal_vy = (-10, 10)

        self.goal_theta = (-10, 10)
        self.goal_omega = (-10, 10)

        ###################### Drone ########################
        self.theta = (0, 2*math.pi)
        self.omega = (-math.pi, math.pi)

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
        self.delta = 0.3

        self.dim_x = [0, 2, 4, 5]
        self.dim_y = [1, 3, 4, 5]
        self.dim_a = [6, 7]

        self.discount = 0.95
        self.threshold = 10
        ############# Discreteness Coefficient ##############
        # 6D state + 2D action
        # (x, y, vx, vy, theta, omega, t1, t2)
        self.ranges = np.array([self.x, self.y, self.vx, self.vy, self.theta, self.omega, self.t1, self.t2])

        # self.step_number = np.array([10, 10, 10, 10, 10, 10, 10, 10])
        # self.step_number = np.array([2, 2, 2, 2, 2, 2, 2, 2]) # used for debug
        self.step_number = np.array([11, 11, 11, 11, 9, 9, 15, 15]) # used for debug
        # self.step_number = np.array([5, 5, 5, 5, 5, 5, 5, 5]) # used for debug
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

        # reward = [none, obstacle or out of range, goal]
        self.reward = np.array([-1, -500, 1000])

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

        self.state_type_x = np.zeros(self.step_number[self.dim_x], dtype = int)
        self.state_type_y = np.zeros(self.step_number[self.dim_y], dtype = int)

    def value_iteration(self):
        # Generate the combination of 4-d data
        # Select the rows which are not in obstacles

        states = np.array(np.meshgrid(self.grid[0],
                                        self.grid[2],
                                        self.grid[4],
                                        self.grid[5])).T.reshape(-1, len(self.dim_x))


        index = []
        for i, s in enumerate(states):
            t = self.state_check(s, self.dim_x)
            if (t != 1):
                index.append(i)

        states = states[index]

        # Update the value array iteratively
        actions = np.array(np.meshgrid(self.grid[6],
                                        self.grid[7])).T.reshape(-1, len(self.dim_a))

        iteration = 0
        transition_count = 0 
        while True:

            delta = 0

            for s in states:
                best_value = 0
                for a in actions:
                    s_ = self.state_transition(s, a)
                    print(s, s_)
                    if (self.check_range(s_, self.dim_x) == 1):
                        reward = self.reward[1]
                    else:
                        s_ = self.seek_nearest_position(s, self.dim_x)
                        reward = self.reward[self.state_check(s_, self.dim_x)]


                    index = self.state_to_index(s, self.dim_x)
                    index_ = self.state_to_index(s_, self.dim_x)
                    print(index, index_)
                    print("______________________________________________")

                    best_value = max(best_value, reward + self.discount * self.value_x[index_[0], index_[1], index_[2], index_[3]])

                    transition_count += 1
                    if (transition_count % 100000 == 0):
                        print(transition_count)

                index = self.state_to_index(s, self.dim_x)
                delta = max(delta, abs(best_value - self.value_x[index[0], index[1], index[2], index[3]]))
                self.value_x[index[0], index[1], index[2], index[3]] = best_value

            if (delta < self.threshold):
                break 

            print("iteraion %d:" %(iteration))
            print(delta)

            self.value_output(iteration, "state")

            iteration += 1


    def value_output(self, iteration, mode = "index"):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_name = "/value_iteration_" + str(iteration) + ".txt"


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

        # temp = (0, 1, 2) -> (none, obstacle / out of range, goal)

        # Check in goal range
        temp = self.check_goal(s, dim)
        if (temp):
            return temp

        # Check in obstacle

        # Get the corresponding dimension of obstacles
        # Due to dimension decomposition
        if (self.obstacles is not None and len(self.obstacles)):

            temp_obs = None
            if (dim[0] == 0): # if it is dim_x
                temp_obs = self.obstacles[:, :2]
            else:
                temp_obs = self.obstacles[:, 2:]

            # Check in obstacle!
            for obs in temp_obs:
                if (s[0] >= obs[0]  and  s[0] <= obs[1]):
                    temp = 1
                    break

            if (temp):
                return temp

        # Check out of range!
        temp = self.check_range(s, dim)

        return temp

    def check_goal(self, s, dim):
        for i, d in enumerate(dim):
            if (s[i] < self.goals[d][0] or s[i] > self.goals[d][1]):
                return 0
        return 2

    def check_range(self, s, dim):
        for i, d in enumerate(dim):
            if (d == 4):
                continue

            if (s[i] < self.ranges[d][0] or s[i] > self.ranges[d][1]):
                return 1

        return 0

    def seek_nearest_position(self, state, dim):
        # This function can find the nearest position on discrete world for one state
        # The type of state is a row of numpy array, e.g. state = np.array([2, 3, 4, 5]) a 4D state
        # The dim stores the corresponding dimension of state. 
        for i in range(len(dim)):
            state[i] = self.grid[dim[i]] [np.argmin(np.absolute(self.grid[dim[i]] - state[i]))]

        return state

    def state_transition(self, state, action):
        # state = [x, vx, theta, omega]

        act = np.sum(action)

        state_ = np.array([state[0] + state[1] * self.delta,
                            state[1] * (1 - self.trans * self.delta / self.mass) - math.sin(state[2]) * self.delta / self.mass * act,
                            state[2] + state[3] * self.delta,
                            state[3] + self.delta / self.inertia * (-self.rot * state[3] - act)])

        if (state_[2] > self.theta[1]):
            state_[2] = self.theta[0] + (state_[2] - self.theta[1])

        if (state_[2] < self.theta[0]):
            state_[2] = self.theta[1] + (state_[2] - self.theta[0])

        return state_


if __name__ == "__main__":
    env = world_env()

    env.add_obstacle(1,3,2,4)
    env.add_obstacle(3,4,3,4)
    env.state_cutting()
    env.state_init()
    env.value_iteration()
    # env.value_output("state")

    # env.state_to_index([1,1,1,1], env.dim_x)
    # print(env.grid[env.dim_x])
    # print(env.state_to_index(np.array([0.3, 0.8, 1, 0.1]), env.dim_x))

    # state = np.array([2, 3, 4, 5])
    # action = np.array([1, 1])
    # print(env.state_transition(state, action))
    

