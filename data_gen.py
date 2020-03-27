import numpy as np
import os
import csv
import time
import sys
print("system path 1:", sys.path)
# sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
# sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')
print("system path 2:", sys.path)
import rospy
from geometry_msgs.msg import Twist, Pose
from geometry_msgs.msg import Wrench
from gazebo_msgs.msg import ModelState
from gazebo_msgs.msg import LinkState
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import GetLinkState
from gazebo_msgs.srv import SetLinkState
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.srv import ApplyBodyWrench
from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import LaserScan

from tf.transformations import euler_from_quaternion, quaternion_from_euler

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matlab.engine
import pickle

from gym_foo.gym_foo.envs.DubinsCarEnv_v0 import DubinsCarEnv_v0

CUR_PATH = os.path.dirname(os.path.abspath(__file__))

def csv_clean(filename, horizon, trajs_type, truncate=False):
    # assert filename == os.environ['PROJ_HOME_3'] + '/data/quad/valFunc_mpc_filled.csv'
    df = pd.read_csv(filename)

    # Use feasible trajectories
    if trajs_type == 'feasible':
        invalidIndices = df[df['col_trajectory_flag'] == 3].index
        df.drop(invalidIndices, inplace=True)
    elif trajs_type == 'infeasible':
        invalidIndices = df[df['col_trajectory_flag'] == 2].index
        df.drop(invalidIndices, inplace=True)
    elif trajs_type == 'all':
        pass
    else:
        raise ValueError("invalid trajs type!")

    # if truncation is required
    if truncate:
        df_truncate = pd.DataFrame(columns=df.columns)
        T = len(df.index)
        for i in range(0, T+1-horizon, horizon):
            for j in range(i, i+horizon):
                tmp = df.iloc[[j]]
                df_truncate = df_truncate.append(tmp)
                # print(tmp['value'].item())
                if tmp['value'].item() in [1000, -400]:
                    # print("value is 1000 or -400")
                    break
        df = df_truncate

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    name_to_save = os.path.splitext(filename)[0] + '_cleaned'
    name_to_save += '_' + trajs_type

    if truncate:
        name_to_save += '_truncated'
    else:
        name_to_save += '_untruncated'
    df.to_csv(name_to_save + '.csv')



""" Be consistent with the saved file name """

def special_func_combine(filenames):
    result = pd.DataFrame()
    for fn in filenames:
        df = pd.read_csv(fn)
        result = pd.concat([result, df], sort=False)
        # df1 = pd.read_csv(filename1)
        # df2 = pd.read_csv(filename2)
        # result = pd.concat([df1, df2], sort=False)

    result.to_csv(os.environ['PROJ_HOME_3']+'/data/quad/valFunc_mpc_filled_final.csv')

def data_balancer():
    fn = os.environ['PROJ_HOME_3']+'/data/quad/valFunc_mpc_filled_final.csv'
    assert os.path.exists(fn)

    df = pd.read_csv(fn)
    T = len(df.index)

    infeasindices = df[df['col_trajectory_flag'] == 3].index
    feasindices   = df[df['col_trajectory_flag'] == 2].index

    infeasNum = len(infeasindices)
    feasNum = len(feasindices)

    print("Number of infeasible data before: ", infeasNum)
    print("Number of feasible data before: ", feasNum)

    if infeasNum < feasNum:
        delindices = np.random.choice(feasindices, np.abs(infeasNum-feasNum), replace=False)
    else:
        delindices = np.random.choice(infeasindices, np.abs(infeasNum-feasNum), replace=False)

    df.drop(delindices, inplace=True)


    infeasindices = df[df['col_trajectory_flag'] == 3].index
    feasindices = df[df['col_trajectory_flag'] == 2].index

    infeasNum = len(infeasindices)
    feasNum = len(feasindices)

    print("Number of infeasible data after: ", infeasNum)
    print("Number of feasible data after: ", feasNum)
    df.to_csv(os.environ['PROJ_HOME_3']+'/data/quad/valFunc_mpc_filled_final.csv')

class Data_Generator(object):
    def __init__(self):
        rospy.init_node("data_collection", anonymous=True, log_level=rospy.INFO)

    def in_obst(self, contact_data):
        if len(contact_data.states) != 0:
            if contact_data.states[0].collision1_name != "" and contact_data.states[0].collision2_name != "":
                return True
        else:
                return False

    def in_goal(self, state, goal_state, goal_torlerance):
        assert len(goal_state) == len(goal_torlerance)
        # {x, y, theta}(dubins' car) or {x, z, phi}(quad)
        if len(goal_state) == 3:
            goal_pos_tolerance = (goal_torlerance[0] + goal_torlerance[1]) / 2
            goal_theta_tolerance = goal_torlerance[2]

            if (np.sqrt((state[0] - goal_state[0]) ** 2 + (state[1] - goal_state[1]) ** 2) <= goal_pos_tolerance
                or (np.abs(state[0] - goal_state[0]) <= goal_pos_tolerance and np.abs(state[1] - goal_state[1]) <= goal_pos_tolerance)) \
                and abs(state[2] - goal_state[2]) < goal_theta_tolerance:
                print("in goal with specific angle!!")
                return True
            else:
                return False
        # {x, y}
        elif len(goal_state) == 2:
            goal_pos_tolerance = (goal_torlerance[0] + goal_torlerance[1]) / 2
            if np.sqrt((state[0] - goal_state[0]) ** 2 + (state[1] - goal_state[1]) ** 2) <= goal_pos_tolerance:
                print("in goal!!")
                return True
            else:
                return False

    def discretize_sensor(self, sensor_data, new_ranges):
        discretized_ranges = []
        full_ranges = float(len(sensor_data.ranges))
        # print("laser ranges num: %d" % full_ranges)

        for i in range(new_ranges):
            new_i = int(i * full_ranges // new_ranges + full_ranges // (2 * new_ranges))
            if sensor_data.ranges[new_i] == float('Inf') or np.isinf(sensor_data.ranges[new_i]):
                discretized_ranges.append(float('Inf'))
            elif np.isnan(sensor_data.ranges[new_i]):
                discretized_ranges.append(float('Nan'))
            else:
                # discretized_ranges.append(int(sensor_data.ranges[new_i]))
                discretized_ranges.append(sensor_data.ranges[new_i])
        return discretized_ranges

    def init_quad(self, x, vx, z, vz, phi, w):
        # init function is for value iteration
        # quad model for value iteration is also 6D, so we need init all six state variables
        quad_pose = Pose()
        quad_pose.position.x = x
        quad_pose.position.y = 0
        quad_pose.position.z = z

        quad_twist = Twist()
        quad_twist.linear.x = vx
        quad_twist.linear.y = 0
        quad_twist.linear.z = vz

        quad_twist.angular.x = 0
        quad_twist.angular.y = w
        quad_twist.angular.z = 0

        qu_x, qu_y, qu_z, qu_w = quaternion_from_euler(0, phi, 0)
        quad_pose.orientation.x = qu_x
        quad_pose.orientation.y = qu_y
        quad_pose.orientation.z = qu_z
        quad_pose.orientation.w = qu_w

        quad_state = ModelState()
        quad_state.model_name = "quadrotor"
        quad_state.pose = quad_pose
        quad_state.twist = quad_twist

        rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)(quad_state)
        return True

    def init_car(self, x, y, theta):
        # init function is for value iteration
        # since car model for value iteration is 3D, so we only initialize x, y, theta
        car_pose = Pose()
        car_pose.position.x = x
        car_pose.position.y = y
        # car_pose.position.z = 0.1

        car_quat_x, car_quat_y, car_quat_z, car_quat_w = quaternion_from_euler(0, 0, theta)
        car_pose.orientation.x = car_quat_x
        car_pose.orientation.y = car_quat_y
        car_pose.orientation.z = car_quat_z
        car_pose.orientation.w = car_quat_w

        car_twist = Twist()
        car_twist.linear.x = 0
        car_twist.linear.y = 0
        car_twist.linear.z = 0

        car_state = ModelState()
        car_state.model_name = 'mobile_base'
        car_state.pose = car_pose
        car_state.twist= car_twist
        rospy.ServiceProxy('gazebo/set_model_state', SetModelState)(car_state)
        return True

    def gen_data(self, horizon, rew_config, data_form='valFunc', agent=None):
        assert agent in ['quad', 'car', 'dubinsCar']
        unfilled_filename = None
        filled_filename = None
        if agent == 'quad':
            unfilled_filename = os.environ['PROJ_HOME_3'] + '/data/quad/' + data_form + '.csv'
            filled_filename   = os.environ['PROJ_HOME_3'] + '/data/quad/' + data_form + '_filled' + '.csv'
            assert os.path.exists(unfilled_filename)

            rews   = None  # only useful when data_form=='valFunc_mpc'
            vpreds = None  # only useful when data_form=='valFunc_mpc'

            """ Specify the associated reward and value from Seth's original mpc data, only when data_form=='valFunc_mpc """
            if data_form == 'valFunc_mpc':
                # cols (1, 2, 3, 4, 5, 6, 13, 14, 15) => {x, vx, z, vz, phi, w, collision_future, collision_curr, col_trajectory_flag}
                raw = np.genfromtxt(unfilled_filename, delimiter=',', skip_header=True,
                                    usecols=(1, 2, 3, 4, 5, 6, 13, 14, 15), dtype=np.float32)


                states = raw[:, :6]
                collision_attr = raw[:, 6:]

                # For quad env: air_space_202002 (more difficult one)
                # goal_state = np.array([4.0, 9.0, 0])  # this is for {x, z, phi}
                # goal_torlerance = np.array([1.0, 1.0, np.pi/6])

                # For quad env: air_space_202002_Francis (a easier one, target angle to the left)
                goal_state = np.array([4.0, 9.0, -np.pi/8])
                goal_torlerance = np.array([1.0, 1.0, np.pi/8])

                # For quad env: air_space_202002_Francis (a easier one, target angle to the right)
                # goal_state = np.array([4.0, 9.0, 0.75])
                # goal_torlerance = np.array([1.0, 1.0, 0.3])

                T = np.shape(states)[0]
                # print("T:", T)
                # T = 1400
                mpc_horizon = horizon
                discount_factor = 0.98

                rews = np.zeros(T, 'float32')
                vpreds = np.zeros(T, 'float32')

                # Data generation using sparse reward setting
                if rew_config == 'sparse':
                    for i in range(mpc_horizon, T + 1, mpc_horizon):
                        for j in reversed(range(i - mpc_horizon, i)):
                            cur_state = states[j, [0,2,4]]
                            if self.in_goal(cur_state, goal_state, goal_torlerance):
                                rews[j] = 1000
                            elif collision_attr[j, 1]:
                                rews[j] = -400
                            else:
                                rews[j] = 0

                            if j == i - 1 or rews[j] == 1000 or rews[j] == -400:
                                vpreds[j] = rews[j]
                            else:
                                vpreds[j] = rews[j] + discount_factor * vpreds[j + 1]

                # Data generation using MPC-like reward setting
                elif rew_config == 'MPC':
                    for i in range(mpc_horizon, T + 1, mpc_horizon):
                        for j in reversed(range(i - mpc_horizon, i)):
                            cur_state = states[j, [0,2,4]]
                            if self.in_goal(cur_state, goal_state, goal_torlerance):
                                rews[j] = 1000
                            elif collision_attr[j, 1]:
                                rews[j] = -400
                            else:
                                delta_x = cur_state[0] - goal_state[0]
                                delta_z = cur_state[1] - goal_state[1]
                                rews[j] = -np.sqrt(delta_x ** 2 + delta_z ** 2)

                            if j == i - 1 or rews[j] == 1000 or rews[j] == -400:
                                vpreds[j] = rews[j]
                            else:
                                vpreds[j] = rews[j] + discount_factor * vpreds[j + 1]
                else:
                    raise ValueError("invalid reward config for generating data!")


                rews = rews.reshape(-1, 1)
                vpreds = vpreds.reshape(-1, 1)

            """ Read unfilled file and choose what we want and write as filled file """
            with open(unfilled_filename, 'r') as csvfile1, open(filled_filename, 'w', newline='') as csvfile2:
                reader = csv.DictReader(csvfile1)
                reader = list(reader)

                """ Prepare fieldnames to write """
                if data_form == 'valFunc_mpc':
                    fieldnames = ['x', 'vx', 'z', 'vz', 'phi', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8',
                                  'reward', 'value', 'cost', 'collision_in_future', 'collision_current', 'col_trajectory_flag']
                elif data_form == 'polFunc':
                    fieldnames = ['x', 'vx', 'z', 'vz', 'phi', 'w', 'a1', 'a2', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
                else:
                    raise ValueError("invalid data form!!")
                writer = csv.DictWriter(csvfile2, fieldnames)
                writer.writeheader()

                """ Run simulator to collect lidar sensor readings """
                rospy.wait_for_service('/gazebo/reset_simulation')
                print("# do I get here??")
                try:
                    rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
                except rospy.ServiceException as e:
                    print("# Reset simulation failed!")

                # print("len of reader:", len(reader))
                for id in range(len(reader)):
                    row = reader[id]
                    x  = float(row['x'])
                    vx = float(row['vx'])
                    z  = float(row['z'])
                    vz = float(row['vz'])
                    phi= float(row['phi'])
                    w  = float(row['w'])
                    # print("current state:", [x,vx,z,vz,phi,w])
                    self.init_quad(x, vx, z, vz, phi, w)

                    sensor_data = None
                    # --- Then take an instant unfreezing ---
                    rospy.wait_for_service('/gazebo/unpause_physics')
                    try:
                        rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
                    except rospy.ServiceException as e:
                        print("/gazebo/unpause_physics service call failed")
                    # --- Receive sensor data ---
                    while sensor_data is None:
                        rospy.wait_for_service("/gazebo/get_model_state")
                        try:
                            sensor_data = rospy.wait_for_message('/scan', LaserScan, timeout=10)
                        except rospy.ServiceException as e:
                            print("/gazebo/get_model_state service call failed!")
                    # --- pause simulation to prepare sample ---
                    rospy.wait_for_service('/gazebo/pause_physics')
                    try:
                        rospy.ServiceProxy('/gazebo/pause_physics', Empty)
                    except rospy.ServiceException as e:
                        print("/gazebo/pause_physics service call failed")
                    discrete_sensor_data = self.discretize_sensor(sensor_data, 8)

                    # --- ignore any invalid sensor readings --- #
                    # if np.isnan(discrete_sensor_data).any() or np.isinf(discrete_sensor_data).any():
                    #     continue

                    if data_form == 'valFunc_mpc':
                        reward = rews[id, -1]
                        value = vpreds[id, -1]
                        cost = float(row['cost'])
                        collision_in_future = float(row['collision_in_future'])
                        collision_current = float(row['collision_current'])
                        col_trajectory_flag = float(row['col_trajectory_flag'])
                        tmp_dict = {'x': x, 'vx': vx, 'z': z, 'vz': vz, 'phi': phi, 'w': w,
                                    'd1': discrete_sensor_data[0],
                                    'd2': discrete_sensor_data[1],
                                    'd3': discrete_sensor_data[2],
                                    'd4': discrete_sensor_data[3],
                                    'd5': discrete_sensor_data[4],
                                    'd6': discrete_sensor_data[5],
                                    'd7': discrete_sensor_data[6],
                                    'd8': discrete_sensor_data[7],
                                    'reward': reward, 'value': value,
                                    'cost': cost, 'collision_in_future': collision_in_future,
                                    'collision_current': collision_current,
                                    'col_trajectory_flag': col_trajectory_flag}
                        # print("tmp_dict:", tmp_dict)
                        assert tmp_dict
                    elif data_form == 'polFunc':
                        a1 = float(row['a1'])
                        a2 = float(row['a2'])
                        tmp_dict = {'x': x, 'vx': vx, 'z': z, 'vz': vz, 'phi': phi, 'w': w, 'a1': a1, 'a2': a2,
                                    'd1': discrete_sensor_data[0],
                                    'd2': discrete_sensor_data[1],
                                    'd3': discrete_sensor_data[2],
                                    'd4': discrete_sensor_data[3],
                                    'd5': discrete_sensor_data[4],
                                    'd6': discrete_sensor_data[5],
                                    'd7': discrete_sensor_data[6],
                                    'd8': discrete_sensor_data[7]}
                        # print("tmp_dict:", tmp_dict)
                        assert tmp_dict
                    else:
                        raise ValueError("invalid data form!!")
                    writer.writerow(tmp_dict)

        elif agent == 'dubinsCar':
            unfilled_filename = os.environ['PROJ_HOME_3'] + '/data/dubinsCar/env_difficult/' + data_form + '.csv'
            filled_filename = os.environ['PROJ_HOME_3'] + '/data/dubinsCar/env_difficult/' + data_form + '_filled' + '.csv'
            assert os.path.exists(unfilled_filename)

            rews   = None  # only useful when data_form=='valFunc_mpc'
            vpreds = None  # only useful when data_form=='valFunc_mpc'

            if data_form == 'valFunc_mpc':
                # now the data reading is using fieldnames with more info (including a lot of new flags)
                # reasons are for mpc with soft-constraints, we use more flags but finally proved not good for initialization.
                # the old valFunc_mpc data with hard constraints do not follow current fieldnames (flags)

                # cols (1, 2, 3, 11, 12) => {x, y, theta, collision_future, collision_curr}
                raw = np.genfromtxt(unfilled_filename, delimiter=',', skip_header=True,
                                       usecols=(1, 2, 3, 11, 12), dtype=np.float32)
                states = raw[:, :3]
                collision_attr = raw[:, 3:]

                goal_state = np.array([3.5, 3.5, np.pi*11/18])
                goal_torlerance = np.array([1.0, 1.0, 0.75])

                T = np.shape(states)[0]
                mpc_horizon = horizon
                discount_factor = 0.95

                rews = np.zeros(T, 'float32')
                vpreds = np.zeros(T, 'float32')

                for i in range(mpc_horizon, T+1, mpc_horizon):
                    for j in reversed(range(i-mpc_horizon, i)):
                        if self.in_goal(states[j, :], goal_state, goal_torlerance):
                            rews[j] = 1000
                        elif collision_attr[j, 1]:
                            rews[j] = -400
                        else:
                            rews[j] = 0

                        if j == i-1 or rews[j] == 1000 or rews[j] == -400:
                            vpreds[j] = rews[j]
                        else:
                            vpreds[j] = rews[j] + discount_factor * vpreds[j+1]
                rews = rews.reshape(-1,1)
                vpreds = vpreds.reshape(-1,1)

            with open(unfilled_filename, 'r') as csvfile1, open(filled_filename, 'w', newline='') as csvfile2:
                """ Read the original unfilled file """
                reader = csv.DictReader(csvfile1)
                reader = list(reader)

                """ Determine the fieldnames to be written on `filled` file """
                if data_form == 'valFunc':
                    fieldnames = ['x', 'y', 'theta', 'value', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
                elif data_form == 'valFunc_mpc':
                    # all field names: x, y, theta, v, w, cost, status, start_in_obstacle, collision_in_trajectory, end_in_target, collision_in_future, collision_current, col_trajectory_flag
                    fieldnames = ['x', 'y', 'theta', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'reward', 'value', 'cost', 'collision_in_future', 'collision_current', 'col_trajectory_flag']
                elif data_form == 'polFunc':
                    fieldnames = ['x', 'y', 'theta', 'vel', 'ang_vel', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
                else:
                    raise ValueError("invalid data form!!")
                writer = csv.DictWriter(csvfile2, fieldnames)
                writer.writeheader()

                """ Run simulator to collect simulated info (laser readings) """
                rospy.wait_for_service('/gazebo/reset_simulation')
                print("# do I get here??")
                try:
                    rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
                except rospy.ServiceException as e:
                    print("# Reset simulation failed!")

                for id in range(len(reader)):
                    row = reader[id]
                    x = float(row['x'])
                    y = float(row['y'])
                    theta = float(row['theta'])
                    # print("current state:", [x, y, theta])
                    # Here since we want sensor data and sensor data is not related to delta and vel
                    self.init_car(x, y, theta)

                    sensor_data = None
                    # --- Then take an instant unfreezing ---
                    rospy.wait_for_service('/gazebo/unpause_physics')
                    try:
                        rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
                    except rospy.ServiceException as e:
                        print("/gazebo/unpause_physics service call failed")

                    # --- Receive sensor data ---
                    while sensor_data is None:
                        rospy.wait_for_service("/gazebo/get_model_state")
                        try:
                            sensor_data = rospy.wait_for_message('/scan', LaserScan, timeout=10)
                        except rospy.ServiceException as e:
                            print("/gazebo/get_model_state service call failed!")

                    # --- pause simulation to prepare sample ---
                    rospy.wait_for_service('gazebo/pause_physics')
                    try:
                        rospy.ServiceProxy('/gazebo/pause_physics', Empty)
                    except rospy.ServiceException as e:
                        print("/gazebo/pause_physics service call failed")
                    discrete_sensor_data = self.discretize_sensor(sensor_data, 8)


                    if data_form == 'valFunc':
                        value = float(row['value'])
                        tmp_dict = {'x': x, 'y': y, 'theta': theta, 'value': value,
                                    'd1': discrete_sensor_data[0],
                                    'd2': discrete_sensor_data[1],
                                    'd3': discrete_sensor_data[2],
                                    'd4': discrete_sensor_data[3],
                                    'd5': discrete_sensor_data[4],
                                    'd6': discrete_sensor_data[5],
                                    'd7': discrete_sensor_data[6],
                                    'd8': discrete_sensor_data[7]}
                    elif data_form == 'valFunc_mpc':
                        reward = rews[id, -1]
                        value  = vpreds[id, -1]
                        cost = float(row['cost'])
                        collision_in_future = float(row['collision_in_future'])
                        collision_current = float(row['collision_current'])
                        col_trajectory_flag = float(row['col_trajectory_flag'])
                        tmp_dict = {'x': x, 'y': y, 'theta': theta,
                                    'd1': discrete_sensor_data[0],
                                    'd2': discrete_sensor_data[1],
                                    'd3': discrete_sensor_data[2],
                                    'd4': discrete_sensor_data[3],
                                    'd5': discrete_sensor_data[4],
                                    'd6': discrete_sensor_data[5],
                                    'd7': discrete_sensor_data[6],
                                    'd8': discrete_sensor_data[7],
                                    'reward': reward, 'value': value,
                                    'cost': cost, 'collision_in_future':collision_in_future,
                                    'collision_current':collision_current,
                                    'col_trajectory_flag':col_trajectory_flag}

                    elif data_form == 'polFunc':
                        vel = float(row['vel'])
                        ang_vel = float(row['ang_vel'])
                        tmp_dict = {'x': x, 'y': y, 'theta': theta, 'vel': vel, 'ang_vel':ang_vel,
                                    'd1': discrete_sensor_data[0],
                                    'd2': discrete_sensor_data[1],
                                    'd3': discrete_sensor_data[2],
                                    'd4': discrete_sensor_data[3],
                                    'd5': discrete_sensor_data[4],
                                    'd6': discrete_sensor_data[5],
                                    'd7': discrete_sensor_data[6],
                                    'd8': discrete_sensor_data[7]}
                    else:
                        raise ValueError("invalid data form!!")

                    assert tmp_dict
                    writer.writerow(tmp_dict)



if __name__ == "__main__":
    data_gen = Data_Generator()
    # data_gen.gen_data(horizon=140, rew_config='sparse', data_form='valFunc_mpc', agent='quad')

    # csv_clean(os.path.join(os.environ['PROJ_HOME_3'], 'data/quad/test_samps_800_N140_warmstart_newangle/valFunc_mpc_filled.csv'),
    #           horizon=140, trajs_type='all', truncate=False)
    # csv_clean(os.path.join(os.environ['PROJ_HOME_3'], 'data/quad/test_samps_800_N140_warmstart_newangle/valFunc_mpc_filled.csv'),
    #           horizon=140, trajs_type='infeasible', truncate=True)
    #


    # filenames = [os.path.join(os.environ['PROJ_HOME_3'], 'data/quad/test_samps_800_N80_warmstart_short_horizon/valFunc_mpc_filled_cleaned_feasible_truncated.csv'),
    #              os.path.join(os.environ['PROJ_HOME_3'], 'data/quad/test_samps_800_N140_warmstart/valFunc_mpc_filled_cleaned_infeasible_truncated.csv'),
    #              os.path.join(os.environ['PROJ_HOME_3'], 'data/quad/test_samps_800_N140_warmstart_newangle/valFunc_mpc_filled_cleaned_infeasible_truncated.csv')]
    # special_func_combine(filenames)

    # data_balancer()