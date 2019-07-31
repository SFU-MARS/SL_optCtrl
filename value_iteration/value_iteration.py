import math
import numpy as np
import pandas as pd



class world_env(object):
    def __init__(self):
        ###################### World Env ####################
        self.x = (0, 5)
        self.y = (0, 5)

        self.gravity = 9.8

        self.obstacle = None

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
        # (x, y, vx, vy, theta, omega, t1, t2)
        # self.step_number = np.array([10, 10, 10, 10, 10, 10, 10, 10])
        self.step_number = np.array([2, 2, 2, 2, 2, 2, 2, 2])
        self.ranges = np.array([self.x, self.y, self.vx, self.vy, self.theta, self.omega, self.t1, self.t2])

        #Grid is used for storing the n-D arrays regard to the discret states after cutting
        self.grid = None

        ############# Initial Points From Goal ##############
        self.state = None



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
        print("theta: ", self.theta)
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

    def state_generator(self, dim = 8):
        dim_x = [0, 2, 4, 5]  
        dim_y = [1, 3, 4, 5]
        dim_a = [6, 7]

        # refer: https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
        self.state = np.array(np.meshgrid(self.grid[0],
                                            self.grid[2],
                                            self.grid[4],
                                            self.grid[5])).T.reshape(-1, len(dim_x))




    def seek_nearest_position(self, state, dim):
        # This function can find the nearest position on discrete world for one state
        # The type of state is a row of numpy array, e.g. state = np.array([2, 3, 4, 5])
        # The dim stores the corresponding dimension of state. 
        for i in range(len(dim)):
            state[i] = self.grid[dim[i]] [np.argmin(np.absolute(self.grid[dim[i]] - state[i]))]

        return state


    

    


if __name__ == "__main__":
    env = world_env()
    env.state_cutting()
    env.state_generator()

    # env.seek_nearest_position(x, dim)
