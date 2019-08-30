import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import gazebo_env
from utils import *
from sensor_msgs.msg import LaserScan

from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist, Pose
from std_srvs.srv import Empty

from gazebo_msgs.srv import GetModelState, SetModelState

import rospy

from tf.transformations import euler_from_quaternion, quaternion_from_euler

# need to be compatitable with model.sdf and world.sdf for custom setting
GOAL_STATE = np.array([3.459, 3.626, 0., 0., 0.])
START_STATE = np.array([-0.182, -3.339, 0., 0., 0.])


class DubinsCarEnv_v0_dqn(gazebo_env.GazeboEnv):
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

        self.num_lasers = 100
        self.state_dim = 5
        self.action_dim = 2

        # the state space has 5 dims: [xpos, ypos, theta, linear_vel(v), angular_vel(w)]
        high_state = np.array([5., 5., np.pi, 2., 0.1])
        # action space has 2 dims: [vel, ang_vel]
        high_action = np.array([2., .5])
        high_obsrv = np.array([5., 5., np.pi, 2., 0.1] + [5 * 2] * self.num_lasers)
        self.state_space = spaces.Box(low=-high_state, high=high_state)
        self.action_space = spaces.Box(low=-high_action, high=high_action)
        self.observation_space = spaces.Box(low=np.array([-5, -5, -np.pi, -2, -0.1] + [0]*self.num_lasers), high=high_obsrv)

        self.goal_state = GOAL_STATE
        self.start_state = START_STATE

        self.control_reward_coff = 0.01
        self.collision_reward = -2*200*self.control_reward_coff*(10**2)
        self.goal_reward = 1000

        self.pre_obsrv = None
        self.reward_type = None
        self.brsEngine = None

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
                discretized_ranges.append(10)
            elif np.isnan(laser_data.ranges[new_i]):
                discretized_ranges.append(0)
            else:
                discretized_ranges.append(int(laser_data.ranges[new_i]))

        return discretized_ranges

    def _in_obst(self, laser_data):

        min_range = 0.5
        for idx, item in enumerate(laser_data.ranges):
            if min_range > laser_data.ranges[idx] > 0:
                return True
        return False

    def _in_goal(self, state):

        assert len(state) == self.state_dim

        x = state[0]
        y = state[1]
        theta = state[2]

        if np.sqrt((x - self.goal_state[0]) ** 2 + (y - self.goal_state[1]) ** 2) <= 1.0:
            return True
        else:
            return False

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

        # velocity, just linear velocity along x-axis
        v = dynamic_data.twist.linear.x
        # angular velocity, just angular velocity along z-axis
        w = dynamic_data.twist.angular.z

        obsrv = [x, y, theta, v, w] + discretized_laser_data

        return obsrv

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_proxy()
            # pose = Pose()
            # pose.position.x = np.random.uniform(low=START_STATE[0] - 0.5, high=START_STATE[0] + 0.5)
            # pose.position.y = np.random.uniform(low=START_STATE[1] - 0.5, high=START_STATE[1] + 0.5)
            # pose.position.z = self.get_model_states(model_name="mobile_base").pose.position.z
            # theta = np.random.uniform(low=START_STATE[2], high=START_STATE[2] + np.pi)
            # ox, oy, oz, ow = quaternion_from_euler(0.0, 0.0, theta)
            # pose.orientation.x = ox
            # pose.orientation.y = oy
            # pose.orientation.z = oz
            # pose.orientation.w = ow
            #
            # reset_state = ModelState()
            # reset_state.model_name = "mobile_base"
            # reset_state.pose = pose
            # self.set_model_states(reset_state)
        except rospy.ServiceException as e:
            print("# Resets the state of the environment and returns an initial observation.")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print ("/gazebo/unpause_physics service call failed")

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
                except rospy.ServiceException as e:
                    print("/gazebo/unpause_physics service call failed")
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print ("/gazebo/pause_physics service call failed")

        obsrv = self.get_obsrv(laser_data, dynamic_data)
        self.pre_obsrv = obsrv

        return np.asarray(obsrv)

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        max_ang_speed = 0.3
        max_linear_speed = 0.5

        # angular_vel = -max_ang_speed + (action // 5) * 0.15
        # linear_vel = -max_linear_speed + (action % 5) * 0.25
        angular_vel = (action - 12) * max_ang_speed * 0.1
        linear_vel = 0.2

        cmd_vel = Twist()
        cmd_vel.linear.x = linear_vel
        cmd_vel.angular.z = angular_vel
        self.vel_pub.publish(cmd_vel)

        laser_data = None
        dynamic_data = None
        while laser_data is None and dynamic_data is None:
            try:
                laser_data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
                # dynamic_data = rospy.wait_for_message('/gazebo/model_states', ModelStates)
                rospy.wait_for_service("/gazebo/get_model_state")
                try:
                    dynamic_data = self.get_model_states(model_name="mobile_base")
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
            reward = 0
        elif self.reward_type == 'ttr' and self.brsEngine is not None:
            # reward = self.brsEngine.evaluate_ttr(np.reshape(obsrv[:5], (1, -1)))
            # reward = 30 / (reward + 0.001)
            # print("reward:", reward)

            ttr = self.brsEngine.evaluate_ttr(np.reshape(obsrv[:5], (1, -1)))
            reward = -ttr
        elif self.reward_type == 'distance':
            reward = -(Euclid_dis((obsrv[0], obsrv[1]), (GOAL_STATE[0], GOAL_STATE[1])))

        done = False
        suc = False

        # 1. when collision happens, done = True
        if self._in_obst(laser_data):
            reward += self.collision_reward
            done = True

        # 2. In the neighbor of goal state, done is True as well. Only considering velocity and pos
        if self._in_goal(np.array(obsrv[:5])):
            reward += self.goal_reward
            done = True
            suc = True

        # 3. Maybe episode length limit is another factor for resetting the robot, stay tuned.
        # waiting to be implemented
        # ---
        # print("reward:", reward)

        return np.asarray(obsrv), reward, done, suc, {}

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
