import numpy as np
import os
import csv
import time

import rospy
from geometry_msgs.msg import Twist, Pose
from geometry_msgs.msg import Wrench
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.srv import ApplyBodyWrench
from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import LaserScan

from tf.transformations import euler_from_quaternion, quaternion_from_euler



import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CUR_PATH = os.path.dirname(os.path.abspath(__file__))

class Data_Generator(object):
    def __init__(self):
        rospy.init_node("quad_shot", anonymous=True, log_level=rospy.INFO)
        self.srv_unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.srv_pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.srv_reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.srv_spawn_model = rospy.ServiceProxy('/gazebo/spawn_model', SpawnModel)
        self.srv_get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.srv_set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    def init_quad(self, srv, x, vx, z, vz, phi, w):
        quad_state = None

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

        srv(quad_state)
        return True

    def in_obst(self, contact_data):
        if len(contact_data.states) != 0:
            if contact_data.states[0].collision1_name != "" and contact_data.states[0].collision2_name != "":
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
                discretized_ranges.append(10)
            elif np.isnan(sensor_data.ranges[new_i]):
                discretized_ranges.append(0)
            else:
                # discretized_ranges.append(int(sensor_data.ranges[new_i]))
                discretized_ranges.append(sensor_data.ranges[new_i])
        return discretized_ranges


    def gen_data(self, num, data_form='valueFunc', use='train'):
        if use == 'train':
            filepath = CUR_PATH + '/data/' + data_form + '_' + use + '.csv'
            # assert not os.path.exists(filepath)
            # num = 35000
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = ['x', 'vx', 'z', 'vz', 'phi', 'w', 'value', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                gMin = np.array([-5, -2, 0, -2, -np.pi, -np.pi])
                gMax = np.array([5, 2, 10, 2, np.pi, np.pi])
                # print("gMin:", gMin)
                # print("gMax:", gMax)

                # --- Start to spawn quadrotor at random loc ---
                # --- First reset Gazebo and freeze ---
                rospy.wait_for_service('/gazebo/reset_simulation')
                print("# do I get here??")
                try:
                    self.srv_reset_proxy()
                except rospy.ServiceException as e:
                    print("# Reset simulation failed!")
                # --- Then do iterations ---
                for i in range(num):
                    print("i:%d" % i)
                    tmp_dict = {}
                    # --- collision checking and resampling if necessary ---
                    while True:
                        sensor_data = None
                        # dynamic_data = None
                        contact_data = None

                        x   = np.random.uniform(low=gMin[0], high=gMax[0])
                        vx  = np.random.uniform(low=gMin[1], high=gMax[1])
                        z   = np.random.uniform(low=gMin[2], high=gMax[2])
                        vz  = np.random.uniform(low=gMin[3], high=gMax[3])
                        phi = np.random.uniform(low=gMin[4], high=gMax[4])
                        w   = np.random.uniform(low=gMin[5], high=gMax[5])
                        value = 0
                        print("initial state:", [x,vx,z,vz,phi,w])
                        self.init_quad(self.srv_set_model_state, x, vx, z, vz, phi, w)

                        # --- Then take an instant unfreezing ---
                        rospy.wait_for_service('/gazebo/unpause_physics')
                        try:
                            self.srv_unpause()
                        except rospy.ServiceException as e:
                            print("/gazebo/unpause_physics service call failed")

                        # --- Receive sensor data ---
                        while sensor_data is None:
                            rospy.wait_for_service("/gazebo/get_model_state")
                            try:
                                sensor_data = rospy.wait_for_message('/scan', LaserScan, timeout=10)
                                # dynamic_data = self.srv_get_model_state(model_name="quadrotor")
                                contact_data = rospy.wait_for_message('/gazebo_ros_bumper', ContactsState, timeout=10)
                            except rospy.ServiceException as e:
                                print("/gazebo/get_model_state service call failed!")
                        # print("state after unfreezing:", dynamic_data)

                        # --- pause simulation to prepare sample ---
                        rospy.wait_for_service('/gazebo/pause_physics')
                        try:
                            self.srv_pause()
                        except rospy.ServiceException as e:
                            print("/gazebo/pause_physics service call failed")

                        discrete_sensor_data = self.discretize_sensor(sensor_data, 8)
                        tmp_dict = {'x': x, 'vx': vx, 'z':z, 'vz':vz, 'phi':phi, 'w':w, 'value':value, \
                                    'd1':discrete_sensor_data[0], 'd2':discrete_sensor_data[1], 'd3':discrete_sensor_data[2], \
                                    'd4':discrete_sensor_data[3], 'd5':discrete_sensor_data[4], 'd6':discrete_sensor_data[5], \
                                    'd7':discrete_sensor_data[6], 'd8':discrete_sensor_data[7]}
                        print("tmp_dict:", tmp_dict)
                        time.sleep(0.5)

                        print("contact_data", contact_data)
                        if not self.in_obst(contact_data) and sensor_data != None:
                            break

                    assert tmp_dict
                    writer.writerow(tmp_dict)


        elif use == 'test':
            assert not os.path.exists(CUR_PATH + '/data/') + data_form + '_' + use
            num = 15000
            pass






if __name__ == "__main__":

    data_gen = Data_Generator()
    data_gen.gen_data(num=7000, data_form='valueFunc', use='train')
