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
        #####################################################

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
        #####################################################

        ############# Discreteness Coefficient ##############
        # 6D state + 2D action
        # (x, y, vx, vy, phi, omega, t1, t2)
        self.step_number = np.array([10, 10, 10, 10, 10, 10, 10, 10])
        self.ranges = np.array([self.x, self.y, self.vx, self.vy, self.phi, self.omega, self.t1, self.t2])
        #####################################################
        self.grid = None
        self.init_point = None
        

    def add_obstacle(self, x1, x2, y1, y2):
        if ((x1 > x2) or (y1 > y2)):
            print("Add obstacle error! Check parameters")
            return

        if (self.obstacle is None):
            self.obstacle = np.array([x1, x2, y1, y2], dtype=float)
        else:
            self.obstacle = numpy.append(self.obstacle, [x1, y1, x2, y2], axis = 0)


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
        self.grid = np.empty((dim, self.step_number.max()))

        for i in range(dim):
            if (self.ranges[i][0] < self.ranges[i][1]):
                self.grid[i] = np.linspace(self.ranges[i][0], self.ranges[i][1], self.step_number[i])
            else:
                print("State {d} range error!".format(i))
                self.grid = None
                break

    def sample_from_goal(self, n_x, n_y, n_vx, n_vy, n_phi, n_omega):
        list_x = np.array([n_x, n_vx, n_phi, n_omega])
        list_y = np.array([n_y, n_vy, n_phi, n_omega])


    


if __name__ == "__main__":
    env = world_env()
    env.add_obstacle(2,3,1,3)
    env.add_obstacle(2,3,1,3)

    # env.world_info()
    env.state_cutting()
    print(env.grid)


    # a = np.linspace(1, 5, 5000000)
    # b = np.linspace(2, 5, 4999999)
    # print(a, b)
    # c = np.array([a, b])
    # import time


    # t1 = time.time()
    # a += 1
    # t2 = time.time()
    # print(t2 - t1)

    # t1 = time.time()
    # c[0] += 1
    # t2 = time.time()
    # print(t2 - t1)

    # print(c[0].dtype)
