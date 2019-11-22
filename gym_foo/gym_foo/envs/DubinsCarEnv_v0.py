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
from gazebo_msgs.msg import ContactsState

import rospy
import sys
print(sys.path)
import tf
print(tf.__file__)
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# same as the goal and start area definition in "playground.world" file
# GOAL_STATE = np.array([3.41924, 3.6939, 0])
GOAL_STATE = np.array([3.5, 3.5, 0])
START_STATE = np.array([-0.222404, -3.27274, 0])

"""
dubins' car 3d model:
x_dot = v * cos(theta)
y_dot = v * sin(theta)
theta_dot = w
"""


class DubinsCarEnv_v0(gazebo_env.GazeboEnv):
    def __init__(self, reward_type, set_additional_goal):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "DubinsCar.launch")

        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self._seed()

        self.num_lasers = 8
        self.state_dim = 3  # x,y,theta
        self.action_dim = 2 # v,w

        high_state = np.array([5.0, 5.0, np.pi])
        low_state  = np.array([-5.0, -5.0, -np.pi])

        high_obsrv = np.concatenate((high_state, np.array([5 * 2] * self.num_lasers)), axis=0)
        low_obsrv  = np.concatenate((low_state,  np.array([0] * self.num_lasers)), axis=0)

        high_action = np.array([2.0, 0.5])
        low_action  = np.array([-2.0, -0.5])


        self.state_space  = spaces.Box(low=-high_state, high=high_state)
        self.observation_space = spaces.Box(low=low_obsrv, high=high_obsrv)
        self.action_space = spaces.Box(low=-high_action, high=high_action)

        self.goal_state  = GOAL_STATE
        self.start_state = START_STATE
        # goal tolerance definition
        self.goal_pos_tolerance = 1.0
        self.goal_theta_tolerance = 0.75

        self.control_reward_coff = 0.01
        self.collision_reward = -2*200*self.control_reward_coff*(10**2)
        self.goal_reward = 1000

        self.pre_obsrv = None
        self.reward_type = reward_type
        self.set_additional_goal = set_additional_goal
        # self.brsEngine = None

        self.step_counter = 0

        print("successfully initialized!!")

    def _discretize_laser(self, laser_data, new_ranges):

        discretized_ranges = []
        full_ranges = float(len(laser_data.ranges))
        # print("laser ranges num: %d" % full_ranges)

        for i in range(new_ranges):
            new_i = int(i * full_ranges // new_ranges + full_ranges // (2 * new_ranges))
            if laser_data.ranges[new_i] == float('Inf') or np.isinf(laser_data.ranges[new_i]):
                # discretized_ranges.append(10.)
                discretized_ranges.append(float('Inf'))
            elif np.isnan(laser_data.ranges[new_i]):
                # discretized_ranges.append(0.)
                discretized_ranges.append(float('Nan'))
            else:
                # discretized_ranges.append(int(laser_data.ranges[new_i]))
                discretized_ranges.append(laser_data.ranges[new_i])

        return discretized_ranges

    def _in_obst(self, laser_data):

        min_range = 0.3
        for idx, item in enumerate(laser_data.ranges):
            if min_range > laser_data.ranges[idx] > 0:
                return True
        return False


    # AMEND: also temporally for DDPG.
    # def _in_obst(self, contact_data):
    #     if len(contact_data.states) != 0:
    #         if contact_data.states[0].collision1_name != "" and contact_data.states[0].collision2_name != "":
    #             return True
    #     return False

    def _in_goal(self, state):

        assert len(state) == self.state_dim

        x = state[0]
        y = state[1]
        theta = state[2]

        if self.set_additional_goal == 'None':
            if np.sqrt((x - self.goal_state[0]) ** 2 + (y - self.goal_state[1]) ** 2) <= self.goal_pos_tolerance:
                print("in goal!!")
                return True
            else:
                return False
        elif self.set_additional_goal == 'angle':
            if np.sqrt((x - self.goal_state[0]) ** 2 + (y - self.goal_state[1]) ** 2) <= self.goal_pos_tolerance \
                and abs(theta - self.goal_state[2]) < self.goal_theta_tolerance:
                print("in goal!!")
                return True
            else:
                return False
        elif self.set_additional_goal == 'vel':
            if np.sqrt((x - self.goal_state[0]) ** 2 + (y - self.goal_state[1]) ** 2) <= self.goal_pos_tolerance \
                and abs(vel - self.goal_state[3]) < 0.25:
                print("in goal!!")
                return True
            else:
                return False
        else:
            raise ValueError("invalid param for set_additional_goal!")

    def get_obsrv(self, laser_data, dynamic_data):

        discretized_laser_data = self._discretize_laser(laser_data, self.num_lasers)

        # here dynamic_data is specially for 'mobile_based' since I specified model name

        # absolute x position
        x = dynamic_data.pose.position.x
        # absolute y position
        y = dynamic_data.pose.position.y
        # heading angle, which == yaw
        ox = dynamic_data.pose.orientation.x
        oy = dynamic_data.pose.orientation.y
        oz = dynamic_data.pose.orientation.z
        ow = dynamic_data.pose.orientation.w
        # axis: sxyz
        _, _, theta = euler_from_quaternion([ox, oy, oz, ow])

        # # velocity, just linear velocity along x-axis
        # v = dynamic_data.twist.linear.x
        # # angular velocity, just angular velocity along z-axis
        # w = dynamic_data.twist.angular.z

        obsrv = [x, y, theta] + discretized_laser_data

        # if any(np.isnan(np.array(obsrv))):
        #     logger.record_tabular("found nan in observation:", obsrv)
        #     obsrv = self.reset()

        return obsrv

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_proxy()
            pose = Pose()
            pose.position.x = np.random.uniform(low=START_STATE[0]-0.5, high=START_STATE[0]+0.5)
            pose.position.y = np.random.uniform(low=START_STATE[1]-0.5, high=START_STATE[1]+0.5)
            pose.position.z = self.get_model_state(model_name="mobile_base").pose.position.z
            theta = np.random.uniform(low=START_STATE[2], high=START_STATE[2]+np.pi)
            ox, oy, oz, ow = quaternion_from_euler(0.0, 0.0, theta)
            pose.orientation.x = ox
            pose.orientation.y = oy
            pose.orientation.z = oz
            pose.orientation.w = ow

            reset_state = ModelState()
            reset_state.model_name = "mobile_base"
            reset_state.pose = pose
            self.set_model_state(reset_state)
        except rospy.ServiceException as e:
            print("# Resets the state of the environment and returns an initial observation.")

        # Unpause simulation only for obtaining observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print ("/gazebo/unpause_physics service call failed")

        # read laser data
        laser_data = None
        dynamic_data = None

        laser_data = rospy.wait_for_message('/scan', LaserScan, timeout=5)

        # Pause simulator to do other operations
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print ("/gazebo/pause_physics service call failed")

        rospy.wait_for_service("/gazebo/get_model_state")
        dynamic_data = self.get_model_state(model_name="mobile_base")

        assert laser_data is not None and dynamic_data is not None
        
        # print("laser data", laser_data)
        # print("dynamics data", dynamic_data)

        obsrv = self.get_obsrv(laser_data, dynamic_data)
        self.pre_obsrv = obsrv

        return np.asarray(obsrv)

    def step(self, action):
        # print("entering to setp func")
        if sum(np.isnan(action)) > 0:
            raise ValueError("Passed in nan to step! Action: " + str(action))

        # action = np.clip(action, -2, 2)

        # [-2, 2] --> [-0.8, 2]
        linear_vel = -0.8 + (2 - (-0.8)) * (action[0] - (-2)) / (2 - (-2))

        # For ppo, do nothing to ang_vel
        angular_vel = action[1]

        # For ddpg, [-2, 2] --> [-0.8, 0.8]
        # angular_vel = -0.8 + (0.8 - (-0.8)) * (action[1] - (-2)) / (2 - (-2))

        # For trpo, clip to [-1,1], then [-1,1] --> [-0.5,0.5]
        # angular_vel = np.clip(action[1], -1, 1)
        # angular_vel = -0.5 + (0.5 - (-0.5)) * (angular_vel - (-1)) / (1 - (-1))

        # print("linear velocity:", linear_vel)
        # print("angular velocity:", angular_vel)

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = angular_vel
        # print("vel_cmd",vel_cmd)
        self.vel_pub.publish(vel_cmd)

        # Unpause simulation only for obtaining observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        laser_data = None
        dynamic_data = None
        # contact_data = None

        # contact_data = rospy.wait_for_message('/gazebo_ros_bumper', ContactsState, timeout=50)
        laser_data = rospy.wait_for_message('/scan', LaserScan, timeout=10)

        # Pause the simulation to do other operations
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        rospy.wait_for_service("/gazebo/get_model_state")
        dynamic_data = self.get_model_state(model_name="mobile_base")

        assert laser_data is not None and dynamic_data is not None


        obsrv = self.get_obsrv(laser_data, dynamic_data)

        # --- special solution for nan/inf observation (especially in case of any invalid sensor readings) --- #
        if any(np.isnan(np.array(obsrv))) or any(np.isinf(np.array(obsrv))):
            logger.record_tabular("found nan or inf in observation:", obsrv)
            obsrv = self.pre_obsrv
            done = True
            self.step_counter = 0

        self.pre_obsrv = obsrv

        assert self.reward_type is not None
        reward = 0

        if self.reward_type == 'hand_craft':
            # reward = 1
            reward += 0
        elif self.reward_type == 'ttr' and self.brsEngine is not None:
            # reward = self.brsEngine.evaluate_ttr(np.reshape(obsrv[:5], (1, -1)))
            # reward = 30 / (reward + 0.001)
            # print("reward:", reward)
            ttr = self.brsEngine.evaluate_ttr(np.reshape(obsrv[:5], (1, -1)))
            reward += -ttr
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
        if self._in_obst(laser_data):
            reward += self.collision_reward
            done = True
            self.step_counter = 0

        # temporary change for ddpg only. For PPO, use things above.
        # if self._in_obst(contact_data):
        #     reward += self.collision_reward
        #     done = True
        #     self.step_counter = 0

        # 2. In the neighbor of goal state, done is True as well. Only considering velocity and pos
        if self._in_goal(np.array(obsrv[:3])):
            reward += self.goal_reward
            done = True
            suc  = True
            self.step_counter = 0

        if self.step_counter >= 300:
            reward += self.collision_reward
            done = True
            self.step_counter = 0

        cur_w = dynamic_data.twist.angular.z
        if cur_w > np.pi:
            done = True
            reward += self.collision_reward / 2

        return np.asarray(obsrv), reward, done, {'suc':suc}

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
