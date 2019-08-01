import math
import numpy as np
import pandas as pd



class world_env(object):
    def __init__(self):
        ###################### World Env ####################
        self.x = (0, 5)
        self.y = (0, 5)

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
        self.theta = (-math.pi, math.pi)
        self.omega = (-math.pi * 2, math.pi * 2)

        self.vx = (-5, 5)
        self.vy = (-5, 5)

        # thrust - action range
        self.t1 = (0, 0.14)
        self.t2 = (0, 0.14)

        self.mass = 1
        self.length = 1
        self.inertia = 1
        self.trans = 1
        self.rot = 1
        self.delta = 1

        ############# Discreteness Coefficient ##############
        # 6D state + 2D action
        # (x, y, vx, vy, theta, omega, t1, t2)
        self.ranges = np.array([self.x, self.y, self.vx, self.vy, self.theta, self.omega, self.t1, self.t2])

        # self.step_number = np.array([10, 10, 10, 10, 10, 10, 10, 10])
        self.step_number = np.array([2, 2, 2, 2, 2, 2, 2, 2]) # used for debug

        # The list of goal dimension
        self.goals = np.array([self.goal_x, self.goal_y, self.goal_vx, self.goal_vy, self.goal_theta, self.goal_omega])

        #Grid is used for storing the n-D arrays regard to the discret states after cutting
        self.grid = None

        ############# The discrete states in the world ##############
        self.state = None
        self.value = None

        self.n_state_type = 3
        self.state_type = None

        # reward = [none, obstacle or out of range, goal]
        self.type_reward = np.array([-1, -500, 1000])

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
        self.grid = np.empty((dim, self.step_number.max()))

        for i in range(dim):
            if (self.ranges[i][0] < self.ranges[i][1]):
                self.grid[i] = np.linspace(self.ranges[i][0], self.ranges[i][1], self.step_number[i])

            # Range exception!
            else:
                print("State %d range error!" %(i))
                self.grid = None
                break

    def state_generator(self, dim = 8):
        dim_x = [0, 2, 4, 5]  
        dim_y = [1, 3, 4, 5]
        dim_a = [6, 7]

        # refer: https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
        self.state = np.array(np.meshgrid(self.grid[0],
                                            self.grid[2],
                                            self.grid[4],
                                            self.grid[5])).T.reshape(-1, len(dim_x))

        self.value = np.zeros(self.state.shape[0])

        self.state_type = np.zeros([self.state.shape[0], self.n_state_type])
        self.state_type = self.state_evaluation(self.state, self.state_type, dim_x)

    def state_evaluation(self, state, state_type, dim):
        for i, s in enumerate(state):
            state_type[i][self.state_check(s, dim)] = 1

        return state_type

    def state_check(self, s, dim):
        # This function is used for returning the state_type of a state
        # Also including check whether the state is out of range

        # temp = (0, 1, 2) -> (none, obstacle, goal)
        temp = 2

        # Check in goal range
        for i, d in enumerate(dim):
            if (s[i] < self.goals[d][0] or s[i] > self.goals[d][1]):
                temp = 0
                break

        if (temp):
            return temp

        # Check in obstacle
        temp_obs = None

        # Get the corresponding dimension of obstacles
        # Due to dimension decomposition
        if (dim[0] == 0):
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

        temp = 0
        # Check out of range!
        for i, d in enumerate(dim):
            if (s[i] < self.ranges[d][0] or s[i] > self.ranges[d][1]):
                temp = 1
                break

        return temp

    def seek_nearest_position(self, state, dim):
        # This function can find the nearest position on discrete world for one state
        # The type of state is a row of numpy array, e.g. state = np.array([2, 3, 4, 5])
        # The dim stores the corresponding dimension of state. 
        for i in range(len(dim)):
            state[i] = self.grid[dim[i]] [np.argmin(np.absolute(self.grid[dim[i]] - state[i]))]

        return state

    def state_transition(self, state, action):
        trans_mat = np.array([1, self.delta, 0, 0],
                             [0, 1 - self.trans * self.delta / self.mass, 0, 0],
                             [0, 0, 1, self.delta],
                             [0, 0, 0, 1 - self.rot * self.delta / self.inertia])

        print(trans_mat)

        # _state = np.matmul(state, trans_mat)
        

    


if __name__ == "__main__":
    env = world_env()

    env.add_obstacle(1,3,2,4)
    env.add_obstacle(3,4,3,4)
    env.state_cutting()
    env.state_generator()    
    
    
    
    

    # TODO List
    # state_check() function test
    # state transition function() implement and test
    # value iteration function() implement and test

