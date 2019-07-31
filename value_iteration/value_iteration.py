import math
import numpy as np
import pandas as pd



class world_env(object):
    def __init__(self):
        ###################### World Env ####################
        self.x = (0, 5)
        self.y = (0, 5)

        self.goal_x = (4, 5)
        self.goal_y = (4, 5)

        self.gravity = 9.8

        self.obstacle = None

        ###################### Drone ########################
        self.phi = (-math.pi, math.pi)
        self.omega = (-math.pi * 2, math.pi * 2)

        self.vx = (0, 1)
        self.vy = (0, 1)

        # thrust - action range
        self.t1 = (0, 0.14)
        self.t2 = (0, 0.14)

        self.mass = 1
        self.length = 1
        self.inertia = 1
        self.trans = 1
        self.rot = 1

        ############# Discreteness Coefficient ##############
        # 6D state + 2D action
        # (x, y, vx, vy, phi, omega, t1, t2)
        self.step_number = np.array([10, 10, 10, 10, 10, 10, 10, 10])
        self.ranges = np.array([self.x, self.y, self.vx, self.vy, self.phi, self.omega, self.t1, self.t2])

        #Grid is used for storing the n-D arrays regard to the discret states after cutting
        self.grid = None


        ############# Initial Points From Goal ##############
        self.init_point = None

        # How many points will sample on each dimension.
        # 6D state + 2D action (n_x, n_y, n_vx, n_vy, n_phi, n_omega, n_t1, n_t2)
        self.sample_number = np.array([2, 2, 2, 2, 2, 2, 2, 2])
        self.goal_ranges = np.array([self.goal_x, self.goal_y, self.vx, self.vy, 
                                    self.phi, self.omega, self.t1, self.t2])

    def add_obstacle(self, x1, x2, y1, y2):
        if ((x1 > x2) or (y1 > y2)):
            print("Add obstacle error! Check parameters")
            return

        if (self.obstacle is None):
            self.obstacle = np.array([[x1, x2, y1, y2]], dtype=float)
        else:
            self.obstacle = np.concatenate((self.obstacle, np.array([[x1, x2, y1, y2]])), axis = 0)

    def world_info(self):
        print("world_x: ", self.x)
        print("world_y: ", self.y)
        print("goal_x: ", self.goal_x)
        print("goal_y: ", self.goal_y)
        print("phi: ", self.phi)
        print("omega: ", self.omega)
        print("step: ", self.step)

        print("\nobstacle: (min_x, min_y, max_x, max_y)")
        for obs in self.obstacle:
            print(obs)

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

    def sample_from_goal(self, dim = 8):
        dim_x = [0, 2, 4, 5]  
        dim_y = [1, 3, 4, 5]
        dim_a = [6, 7]

        self.init_point = np.empty((dim, self.sample_number.max()))

        for i in range(dim):
            if (self.goal_ranges[i][0] < self.goal_ranges[i][1]):
                self.init_point[i] = np.linspace(self.goal_ranges[i][0], 
                                                self.goal_ranges[i][1], 
                                                self.sample_number[i]) 

        # refer: https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
        self.init_point = np.array(np.meshgrid(self.init_point[0],
                                                self.init_point[2],
                                                self.init_point[4],
                                                self.init_point[5])).T.reshape(-1, len(dim_x))

    def find_nearest_position(self, state, dim):
        # This function can find the nearest position on discrete world for one state
        # The type of state is a row of numpy array, e.g. state = np.array([2, 3, 4, 5])
        # The dim stores the corresponding dimension of state. 
        for i in range(len(dim)):
            state[i] = self.grid[dim[i]] [np.argmin(np.absolute(self.grid[dim[i]] - state[i]))]

        return state


    def previous_state_generator(self):
        # print(self.grid)
        return

    


if __name__ == "__main__":
    env = world_env()
    env.state_cutting()
    # env.sample_from_goal()

    x = np.array([4.2, 4.8])
    dim = [0, 1]

    env.previous_state_generator()
    env.find_nearest_position(x, dim)
