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

# need to be compatitable with model.sdf and world.sdf for custom setting
# GOAL_STATE = np.array([3.459, 3.626, 0., 0., 0.])
GOAL_STATE = np.array([3.459, 3.626, 0.75, 0., 0.])
START_STATE = np.array([-0.182, -3.339, 0., 0., 0.])


class DubinsCarEnv_v0(gazebo_env.GazeboEnv):
    def __init__(self):
        # Launch the simulation with the given launchfile name
        # Notice： here we use 5D state. But in DubinsCarEngine we will use 3D state. We need seperate these two parts to make program more flexiable
        gazebo_env.GazeboEnv.__init__(self, "DubinsCarCircuitGround_v0.launch")

        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.get_model_states = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_model_states = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self._seed()

        self.laser_num = 8
        self.state_dim = 5
        self.action_dim = 2

        high_state = np.array([5., 5., np.pi, 2., 0.5])
        high_action = np.array([2., .5])
        high_obsrv = np.array([5., 5., np.pi, 2., 0.5] + [5 * 2] * self.laser_num)
        self.state_space = spaces.Box(low=-high_state, high=high_state)
        self.action_space = spaces.Box(low=-high_action, high=high_action)
        self.observation_space = spaces.Box(low=np.array([-5, -5, -np.pi, -2, -0.5] + [0]*self.laser_num), high=high_obsrv)

        self.goal_state = GOAL_STATE
        self.start_state = START_STATE

        self.control_reward_coff = 0.01
        self.collision_reward = -2*200*self.control_reward_coff*(10**2)
        self.goal_reward = 1000

        self.pre_obsrv = None
        self.reward_type = None
        self.set_additional_goal = None
        self.brsEngine = None

        self.step_counter = 0

        print("successfully initialized!!")

    def _discretize_laser(self, laser_data, new_ranges):

        discretized_ranges = []
        # mod = int(len(laser_data.ranges)/new_ranges)
        # for i, item in enumerate(laser_data.ranges):
        #     if (i+1) % mod == 0:
        #         if laser_data.ranges[i] == float('Inf') or np.isinf(laser_data.ranges[i]):
        #             discretized_ranges.append(10)
        #         elif np.isnan(laser_data.ranges[i]):
        #             discretized_ranges.append(0)
        #         else:
        #             discretized_ranges.append(int(laser_data.ranges[i]))

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

    def _in_obst(self, laser_data):

        min_range = 0.4
        for idx, item in enumerate(laser_data.ranges):
            if min_range > laser_data.ranges[idx] > 0:
                return True
        return False
    # def _in_obst(self, contact_data):
    #     if len(contact_data.states) != 0:
    #         if contact_data.states[0].collision1_name != "" and contact_data.states[0].collision2_name != "":
    #             return True
    #     else:
    #             return False

    def _in_goal(self, state):

        assert len(state) == self.state_dim

        x = state[0]
        y = state[1]
        theta = state[2]
        vel = state[3]

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
                and abs(vel - self.goal_state[3]) < 0.25:
                print("in goal!!")
                return True
            else:
                return False
        else:
            raise ValueError("invalid param for set_additional_goal!")

    def get_obsrv(self, laser_data, dynamic_data):

        discretized_laser_data = self._discretize_laser(laser_data, self.laser_num)

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

        # velocity, just linear velocity along x-axis
        v = dynamic_data.twist.linear.x
        # angular velocity, just angular velocity along z-axis
        w = dynamic_data.twist.angular.z

        obsrv = [x, y, theta, v, w] + discretized_laser_data

        if any(np.isnan(np.array(obsrv))):
            logger.record_tabular("found nan in observation:", obsrv)
            obsrv = self.reset()

        return obsrv

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
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
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print ("/gazebo/unpause_physics service call failed")
        
        # print("successfully unpaused!!")
        
        #laser_data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
        
        # read laser data
        laser_data = None
        dynamic_data = None

        while laser_data is None and dynamic_data is None:
            try:
                laser_data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
                # dynamic_data = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=5)
                rospy.wait_for_service("/gazebo/get_model_state")
                try:
                    dynamic_data = self.get_model_states(model_name="mobile_base")
                    # dynamic_data = self.get_model_states(model_name="dubins_car")
                except rospy.ServiceException as e:
                    print("/gazebo/unpause_physics service call failed")
            except:
                pass
        
        # print("laser data", laser_data)
        # print("dynamics data", dynamic_data)
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print ("/gazebo/pause_physics service call failed")

        obsrv = self.get_obsrv(laser_data, dynamic_data)
        self.pre_obsrv = obsrv
        # print("successfully reset!!")
        return np.asarray(obsrv)

    def step(self, action):
        
        # print("entering to setp func")
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        if sum(np.isnan(action)) > 0:
            raise ValueError("Passed in nan to step! Action: " + str(action))

        # linear_acc = action[0] + 1.0
        # angular_acc = action[1]
        #
        # if sum(np.isnan(action)) > 0:
        #     raise ValueError("Passed in nan to step! Action: " + str(action))
        #
        # pre_phi = self.pre_obsrv[2]
        #
        # cmd_vel = Twist()
        # cmd_vel.linear.x = self.pre_obsrv[3] + linear_acc * np.cos(pre_phi)
        # cmd_vel.angular.z = self.pre_obsrv[4] + angular_acc
        # self.vel_pub.publish(cmd_vel)

        # linear_acc = action[0]
        # angular_acc = action[1]
        #
        # linear_vel = self.pre_obsrv[3] + linear_acc
        # angular_vel = self.pre_obsrv[4] + angular_acc
        #
        # # clip angular velocity, from (-0.33 to + 0.33)
        # max_ang_speed = 0.3
        # angular_vel = (angular_vel-10)*max_ang_speed*0.1
        #
        # cmd_vel = Twist()
        # cmd_vel.linear.x = linear_vel
        # cmd_vel.angular.z = angular_vel
        # self.vel_pub.publish(cmd_vel)

        # print("linear_vel:", linear_vel)
        # print("angular_vel:", angular_vel)

        # max_linear_vel = 0.5
        # linear_vel = (linear_vel - 10) * max_linear_vel * 0.1

        # clip angular velocity, from (-1.0 to + 1.0)
        # max_ang_vel = 0.3
        # angular_vel = (angular_vel - 10) * max_ang_vel * 0.1

        # if angular_vel > self.action_space.high[1]:
        #     angular_vel = self.action_space.high[1]
        #
        # if angular_vel < self.action_space.low[1]:
        #     angular_vel = self.action_space.low[1]

        # cur_state = ModelState()
        # cur_state.model_name = "dubins_car"
        # cur_state.twist = cmd_vel
        #
        # cur_state.pose.position.x = 0
        # cur_state.pose.position.y = 0
        #
        # cur_state.reference_frame = "dubins_car"
        # self.set_model_states(cur_state)


        # print("action:", action)
        # action[0] + 0.3 if action[0] > 0 else action[0] - 0.3


        # [-2, 2] --> [-0.8, 2]
        linear_vel = -0.8 + (2 - (-0.8)) * (action[0] - (-2)) / (2 - (-2))
        # [-2,2] --> [-0.8, 0.8]
        # For ddpg, use this line below
        # angular_vel = -0.8 + (0.8 - (-0.8)) * (action[1] - (-2)) / (2 - (-2))
        angular_vel = action[1]

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = angular_vel
        # print("vel_cmd",vel_cmd)

        self.vel_pub.publish(vel_cmd)


        laser_data = None
        dynamic_data = None
        # contact_data = None
        while laser_data is None and dynamic_data is None:
            try:
                laser_data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
                # dynamic_data = rospy.wait_for_message('/gazebo/model_states', ModelStates)
                rospy.wait_for_service("/gazebo/get_model_state")
                try:
                    # contact_data = rospy.wait_for_message('/gazebo_ros_bumper', ContactsState, timeout=50)
                    dynamic_data = self.get_model_states(model_name="mobile_base")
                    # dynamic_data = self.get_model_states(model_name="dubins_car")
                except rospy.ServiceException as e:
                    print("/gazebo/unpause_physics service call failed")
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        obsrv = self.get_obsrv(laser_data, dynamic_data)
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
        if self._in_goal(np.array(obsrv[:5])):
            reward += self.goal_reward
            done = True
            suc  = True
            self.step_counter = 0

        if self.step_counter >= 300:
            reward += self.collision_reward
            done = True
            self.step_counter = 0

        if obsrv[4] > np.pi * 2:
            done = True
            reward += self.collision_reward / 2

        # 3. Maybe episode length limit is another factor for resetting the robot, stay tuned.
        # waiting to be implemented
        # ---
        # print("reward:", reward)
        return np.asarray(obsrv), reward, done, suc, {}

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
