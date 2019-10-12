import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import gazebo_env
from utils import *

from utils.utils import *

from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist, Pose, Pose2D
from sensor_msgs.msg import LaserScan

from std_srvs.srv import Empty
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.srv import GetLinkState, SetLinkState
from gazebo_msgs.msg import ContactsState
from ackermann_msgs.msg import AckermannDriveStamped

import rospy
import sys
print(sys.path)
import tf
print(tf.__file__)
from tf.transformations import euler_from_quaternion, quaternion_from_euler




# For ackermann bicycle model, the dynamics is 5D.
# x, y, theta, steering_angle
# GOAL_STATE = ...
START_STATE = np.array([4.2422, 0, 1.5804, 0, 0])


class AckermannEnv_v0(gazebo_env.GazeboEnv):
    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "DubinsCarCircuitGround_v0.launch")

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.get_model_states = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_model_states = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        srv_get_link_state = rospy.ServiceProxy('/ackermann_vehicle/gazebo/get_link_state', GetLinkState)
        srv_set_link_state = rospy.ServiceProxy('/ackermann_vehicle/gazebo/set_link_state', SetLinkState)

        ack_publisher = rospy.Publisher('/ackermann_vehicle/ackermann_cmd', AckermannDriveStamped, queue_size=5)

        self._seed()

        self.laser_num = 8
        self.state_dim = 5
        self.action_dim = 2

        # high_state = np.array([5., 5., np.pi, 2., 0.5])
        # high_action = np.array([2., .5])
        # high_obsrv = np.array([5., 5., np.pi, 2., 0.5] + [5 * 2] * self.laser_num)
        # self.state_space = spaces.Box(low=-high_state, high=high_state)
        # self.action_space = spaces.Box(low=-high_action, high=high_action)
        # self.observation_space = spaces.Box(low=np.array([-5, -5, -np.pi, -2, -0.5] + [0]*self.laser_num), high=high_obsrv)

        self.goal_state = GOAL_STATE
        self.start_state = START_STATE

        self.control_reward_coff = 0.01
        self.collision_reward = -2*200*self.control_reward_coff*(10**2)
        self.goal_reward = 1000

        self.pre_obsrv = None
        self.reward_type = None
        self.set_additional_goal = None
        # self.brsEngine = None
        self.vf_load = False
        self.pol_load = False

        self.step_counter = 0

        print("successfully initialized!!")

    def _discretize_laser(self, laser_data, new_ranges):

        discretized_ranges = []
        full_ranges = float(len(laser_data.ranges))
        # print("laser ranges num: %d" % full_ranges)

        for i in range(new_ranges):
            new_i = int(i * full_ranges // new_ranges + full_ranges // (2 * new_ranges))
            if laser_data.ranges[new_i] == float('Inf') or np.isinf(laser_data.ranges[new_i]):
                discretized_ranges.append(10.)
            elif np.isnan(laser_data.ranges[new_i]):
                discretized_ranges.append(0.)
            else:
                discretized_ranges.append(int(laser_data.ranges[new_i]))

        return discretized_ranges

    # def _in_obst(self, laser_data):
    #
    #     min_range = 0.4
    #     for idx, item in enumerate(laser_data.ranges):
    #         if min_range > laser_data.ranges[idx] > 0:
    #             return True
    #     return False

    def _in_obst(self, contact_data):
        if len(contact_data.states) != 0:
            if contact_data.states[0].collision1_name != "" and contact_data.states[0].collision2_name != "":
                return True
        else:
                return False

    def _in_goal(self, state):

        assert len(state) == self.state_dim

        x = state[0]
        y = state[1]
        theta = state[2]
        vel = state[3]
        steering_angle = state[4]

        # In ackermann bicycle model, velocity is no longer part of state, so we remove the option of set_additional_goal as 'vel'
        if self.set_additional_goal == 'None':
            if np.sqrt((x - self.goal_state[0]) ** 2 + (y - self.goal_state[1]) ** 2) <= 1.0:
                print("in goal!!")
                return True
            else:
                return False
        elif self.set_additional_goal == 'angle':
            if np.sqrt((x - self.goal_state[0]) ** 2 + (y - self.goal_state[1]) ** 2) <= 1.0 \
                and abs(theta - self.goal_state[2]) < 0.40:
                print("in goal!!")
                return True
            else:
                return False
        elif self.set_additional_goal == 'vel':
            if np.sqrt((x - self.goal_state[0]) ** 2 + (y - self.goal_state[1]) ** 2) <= 1.0 \
                and abs(vel - self.goal_state[3]) < 0.40:
                print("in goal!!")
                return True
            else:
                return False
        else:
            raise ValueError("invalid param for set_additional_goal!")

    def get_obsrv(self, laser_data, model_dys, lfw_dys, rfw_dys):

        # Note that here in order to get steering angle as state, we need the link state of left front wheel and right front wheel.
        # lfw_dys: left front wheel
        # rfw_dys: right front wheel
        # base_link_dys: base link

        discretized_laser_data = self._discretize_laser(laser_data, self.laser_num)

        # ------ Proceed on whole model -------- #
        # absolute x position
        x = model_dys.pose.position.x
        # absolute y position
        y = model_dys.pose.position.y
        # heading angle, which == yaw
        ox = model_dys.pose.orientation.x
        oy = model_dys.pose.orientation.y
        oz = model_dys.pose.orientation.z
        ow = model_dys.pose.orientation.w
        # axis: sxyz
        _, _, theta = euler_from_quaternion([ox, oy, oz, ow])
        # velocity, just linear velocity along x-axis
        v = model_dys.twist.linear.x

        # ------- Proceed on left front wheel link ------- #
        lox = lfw_dys.link_state.pose.orientation.x
        loy = lfw_dys.link_state.pose.orientation.y
        loz = lfw_dys.link_state.pose.orientation.z
        low = lfw_dys.link_state.pose.orientation.w
        _, _, lyaw = euler_from_quaternion([lox, loy, loz, low])

        # ------- Proceed on right front wheel link ------- #
        rox = rfw_dys.link_state.pose.orientation.x
        roy = rfw_dys.link_state.pose.orientation.y
        roz = rfw_dys.link_state.pose.orientation.z
        row = rfw_dys.link_state.pose.orientation.w
        _, _, ryaw = euler_from_quaternion([rox, roy, roz, row])

        if lyaw > 1:
            lyaw = lyaw - np.pi
        elif lyaw < -1:
            lyaw = lyaw + np.pi

        if ryaw > 1:
            ryaw = ryaw - np.pi
        elif ryaw < -1:
            ryaw = ryaw + np.pi

        delta = (lyaw + ryaw) / 2

        obsrv = [x, y, theta, v, delta] + discretized_laser_data

        if any(np.isnan(np.array(obsrv))):
            logger.record_tabular("found nan in observation:", obsrv)
            obsrv = self.reset()

        return obsrv

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/ackermann_vehicle/gazebo/reset_simulation')
        try:
            self.reset_proxy()
            pose = Pose()
            pose.position.x = np.random.uniform(low=START_STATE[0]-0.5, high=START_STATE[0]+0.5)
            pose.position.y = np.random.uniform(low=START_STATE[1]-0.5, high=START_STATE[1]+0.5)
            pose.position.z = self.get_model_states(model_name="mobile_base").pose.position.z
            theta = np.random.uniform(low=START_STATE[2], high=START_STATE[2]+np.pi)
            ox, oy, oz, ow = quaternion_from_euler(0.0, 0.0, theta)
            pose.orientation.x = ox
            pose.orientation.y = oy
            pose.orientation.z = oz
            pose.orientation.w = ow

            reset_state = ModelState()
            reset_state.model_name = "mobile_base"
            reset_state.pose = pose
            self.set_model_states(reset_state)
        except rospy.ServiceException as e:
            print("# Resets the state of the environment and returns an initial observation.")

        # Unpause simulation to make observation
        rospy.wait_for_service('/ackermann_vehicle/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print ("# /ackermann_vehicle/gazebo/unpause_physics service call failed")

        laser_data = None
        contact_data = None
        model_dys = None
        lfw_dys = None
        rfw_dys = None

        while laser_data is None or model_dys is None or lfw_dys is None or rfw_dys is None:
            # ----- obtain laser and contact data -----
            try:
                laser_data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
                contact_data = rospy.wait_for_message('/gazebo_ros_bumper', ContactsState, timeout=50)
            except ROSException as e:
                print("# laser data return failed")

            # ------ obtain whole model dys data -------
            rospy.wait_for_service("/ackermann_vehicle/gazebo/get_model_state")
            try:
                model_dys = self.get_model_states(model_name="ackermann_vehicle")
            except rospy.ServiceException as e:
                print("/ackermann_vehicle/gazebo/get_model_states service call failed")

            # ------ obtain link states of left front wheel and right front wheel ------
            rospy.wait_for_service("/ackermann_vehicle/gazebo/get_link_state")
            try:
                lfw_dys = srv_get_link_state(link_name='left_front_wheel', reference_frame='base_link')
                rfw_dys = srv_get_link_state(link_name='right_front_wheel', reference_frame='base_link')
            except rospy.ServiceException as e:
                print("# /ackermann_vehicle/gazebo/get_link_state service call failed")
        
        # print("laser data", laser_data)
        rospy.wait_for_service('/ackermann_vehicle/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print ("/ackermann_vehicle/gazebo/pause_physics service call failed")

        obsrv = self.get_obsrv(laser_data, model_dys, lfw_dys, rfw_dys)
        self.pre_obsrv = obsrv

        return np.asarray(obsrv)

    def step(self, action):
        
        # print("entering to setp func")
        rospy.wait_for_service('/ackermann_vehicle/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/ackermann_vehicle/gazebo/unpause_physics service call failed")

        if sum(np.isnan(action)) > 0:
            raise ValueError("Passed in nan to step! Action: " + str(action))

        # control #1: acceleration: [-2, 2] --> [-1, 1]
        acc = -1.0 + (1 - (-1.0)) * (action[0] - (-2)) / (2 - (-2))
        # control #2: steering angle rate: [-2,2] --> [-0.5, 0.5]
        steering_rate = -0.5 + (0.5 - (-0.5)) * (action[1] - (-2)) / (2 - (-2))

        ack_msg = AckermannDriveStamped()
        # ack_msg.header.stamp = rospy.Time.now()
        # ack_msg.header.frame_id = ''
        ack_msg.drive.steering_angle = 0.75 if steering_rate >= 0 else -0.75
        ack_msg.drive.steering_angle_velocity = steering_rate
        ack_msg.drive.speed = 5.0 if acc >= 0 else -5.0
        ack_publisher.publish(ack_msg)


        laser_data = None
        contact_data = None
        model_dys = None
        lfw_dys = None
        rfw_dys = None
        while laser_data is None or model_dys is None or lfw_dys is None or rfw_dys is None:
            # ----- obtain laser data and contact data -----
            try:
                laser_data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
                contact_data = rospy.wait_for_message('/gazebo_ros_bumper', ContactsState, timeout=50)
            except ROSException as e:
                print("# laser data return failed")

            # ------ obtain whole model dys data -------
            rospy.wait_for_service("/ackermann_vehicle/gazebo/get_model_state")
            try:
                model_dys = self.get_model_states(model_name="ackermann_vehicle")
            except rospy.ServiceException as e:
                print("/ackermann_vehicle/gazebo/get_model_states service call failed")

            # ------ obtain link states of left front wheel and right front wheel ------
            rospy.wait_for_service("/ackermann_vehicle/gazebo/get_link_state")
            try:
                lfw_dys = srv_get_link_state(link_name='left_front_wheel', reference_frame='base_link')
                rfw_dys = srv_get_link_state(link_name='right_front_wheel', reference_frame='base_link')
            except rospy.ServiceException as e:
                print("# /ackermann_vehicle/gazebo/get_link_state service call failed")

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("# /ackermann_vehicle/gazebo/pause_physics service call failed")

        obsrv = self.get_obsrv(laser_data, model_dys, lfw_dys, rfw_dys)
        self.pre_obsrv = obsrv

        assert self.reward_type is not None
        reward = 0

        # Here no ttr reward at all.
        if self.reward_type == 'hand_craft':
            # reward = 1
            reward += 0
        elif self.reward_type == 'distance':
            reward += -(Euclid_dis((obsrv[0], obsrv[1]), (GOAL_STATE[0], GOAL_STATE[1])))
        elif self.reward_type == 'distance_lambda_0.1':
            delta_x = obsrv[0] - GOAL_STATE[0]
            delta_y = obsrv[1] - GOAL_STATE[1]
            delta_theta = obsrv[2] - GOAL_STATE[2]

            reward += -np.sqrt(delta_x**2 + delta_y**2 + 0.1 * delta_theta ** 2)
        elif self.reward_type == 'distance_lambda_1':
            delta_x = obsrv[0] - GOAL_STATE[0]
            delta_y = obsrv[1] - GOAL_STATE[1]
            delta_theta = obsrv[2] - GOAL_STATE[2]

            reward += -np.sqrt(delta_x**2 + delta_y**2 + 1.0 * delta_theta ** 2)
        elif self.reward_type == 'distance_lambda_10':
            delta_x = obsrv[0] - GOAL_STATE[0]
            delta_y = obsrv[1] - GOAL_STATE[1]
            delta_theta = obsrv[2] - GOAL_STATE[2]

            reward += -np.sqrt(delta_x**2 + delta_y**2 + 10.0 * delta_theta ** 2)
        else:
            raise ValueError("no option for step reward!")

        done = False
        suc  = False
        self.step_counter += 1
        # print("step reward:", reward)

        # 1. when collision happens, done = True
        # if self._in_obst(laser_data):
        #     reward += self.collision_reward
        #     done = True
        #     self.step_counter = 0

        if self._in_obst(contact_data):
            reward += self.collision_reward
            done = True
            self.step_counter = 0

        # 2. In the neighbor of goal state, done is True as well. Only considering velocity and pos
        if self._in_goal(np.array(obsrv[:5])):
            reward += self.goal_reward
            done = True
            suc  = True
            self.step_counter = 0

        # 3. Maybe episode length limit is another factor for resetting the robot, stay tuned.
        if self.step_counter >= 100:
            reward += self.collision_reward
            done = True
            self.step_counter = 0


        return np.asarray(obsrv), reward, done, suc, {}

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
