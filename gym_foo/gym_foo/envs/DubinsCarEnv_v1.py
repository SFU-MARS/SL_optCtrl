import gym
from gym import spaces
from gym.utils import seeding

from utils.tools import *
import utils.logger as logger
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan

from std_srvs.srv import Empty
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ContactsState

import subprocess
import rospy
import sys
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler



GOAL_STATE = np.array([3.5, 3.5, np.pi*11/18, 0, 0])    # around 110 degrees as angle
START_STATE = np.array([-0.222404, -3.27274, 0, 0, 0])


"""
dubins' car 5d model:
x_dot = v * cos(theta)
y_dot = v * sin(theta)
theta_dot = w
v_dot = a_v
w_dot = a_w
"""

print("I'm running DubinsCarEnv_v1 (5d dynamics) with subscriber ...")
class DubinsCarEnv_v1(gym.Env):
    def __init__(self, reward_type="hand_craft", set_additional_goal="None"):
        # Build a ros node
        rospy.init_node('DubinsCarEnv', anonymous=True)

        # Define necessary ros services
        self.vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # Define ros subscriber for receiving sensor readings. Note: this ways is more stable than using "wait_for_message"
        rospy.Subscriber("/scan", LaserScan, self._laser_scan_callback)
        rospy.Subscriber("/gazebo_ros_bumper", ContactsState, self._contact_callback)
        self.laser_data = LaserScan()
        self.contact_data = ContactsState()

        # Task-specific settings
        self.num_lasers = 8
        self.state_dim  = 5  # x,y,theta,v,w
        self.action_dim = 2  # vel_acc, ang_acc

        high_state = np.array([5.0, 5.0, np.pi, 5.0, np.pi*5])
        low_state  = np.array([-5.0, -5.0, -np.pi, -5.0, -np.pi*5])

        high_obsrv = np.concatenate((high_state, np.array([5 * 2] * self.num_lasers)), axis=0)
        low_obsrv  = np.concatenate((low_state,  np.array([0] * self.num_lasers)), axis=0)

        # This is consistent with the step() function clipping range.
        # Since you cannot expect NN output is in line with your real physics property. So there always needs transformation
        high_action = np.array([2.0, 2.0])
        low_action  = np.array([-2.0, -2.0])

        self.state_space  = spaces.Box(low=low_state, high=high_state)
        self.observation_space = spaces.Box(low=low_obsrv, high=high_obsrv)
        self.action_space = spaces.Box(low=low_action, high=high_action)

        self.goal_state  = GOAL_STATE
        self.start_state = START_STATE

        # goal tolerance definition
        self.goal_pos_tolerance = 1.0
        self.goal_theta_tolerance = 0.75  # around 45 degrees

        self.control_reward_coff = 0.01
        self.collision_reward = -2 * 200 * self.control_reward_coff*(10**2)
        self.goal_reward = 1000

        self.pre_obsrv  = None
        self.pre_vel    = None
        self.pre_angvel = None
        self.reward_type = reward_type
        self.set_additional_goal = set_additional_goal

        self.step_counter = 0
        self._max_episode_steps = 300
        logger.log("DubinsCarEnv-v1 is successfully initialized!!")

    def _laser_scan_callback(self, laser_msg):
        self.laser_data = laser_msg

    def _contact_callback(self, contact_msg):
        self.contact_data = contact_msg

    def get_laser(self):
        return self.laser_data

    def get_contact(self):
        return self.contact_data

    def _discretize_laser(self, laser_data, new_ranges):

        discretized_ranges = []
        full_ranges = float(len(laser_data.ranges))

        for i in range(new_ranges):
            new_i = int(i * full_ranges // new_ranges + full_ranges // (2 * new_ranges))
            if laser_data.ranges[new_i] == float('Inf') or np.isinf(laser_data.ranges[new_i]):
                discretized_ranges.append(float('Inf'))
            elif np.isnan(laser_data.ranges[new_i]):
                discretized_ranges.append(float('Nan'))
            else:
                discretized_ranges.append(laser_data.ranges[new_i])

        return discretized_ranges

    # This function normally gets used under PPO algorithm
    # def _in_obst(self, laser_data):
    #
    #     min_range = 0.3
    #     for idx, item in enumerate(laser_data.ranges):
    #         if min_range > laser_data.ranges[idx] > 0:
    #             return True
    #     return False


    # added by XLV: temporally for DDPG.
    def _in_obst(self, contact_data):
        if len(contact_data.states) != 0:
            if contact_data.states[0].collision1_name != "" and contact_data.states[0].collision2_name != "":
                return True
        return False


    def _in_goal(self, state):

        assert len(state) == self.state_dim

        x = state[0]
        y = state[1]
        theta = state[2]

        vel = 0  # not used any more

        if self.set_additional_goal == 'None':
            if np.sqrt((x - self.goal_state[0]) ** 2 + (y - self.goal_state[1]) ** 2) <= self.goal_pos_tolerance:
                logger.log("in goal!!")
                return True
            else:
                return False
        elif self.set_additional_goal == 'angle':
            if np.sqrt((x - self.goal_state[0]) ** 2 + (y - self.goal_state[1]) ** 2) <= self.goal_pos_tolerance \
                and abs(theta - self.goal_state[2]) < self.goal_theta_tolerance:
                logger.log("in goal with specific angle!!")
                logger.log("theta:%f" % theta)
                logger.log("goal theta range from %f to %f" % ((GOAL_STATE[2] - self.goal_theta_tolerance), (GOAL_STATE[2] + self.goal_theta_tolerance)))
                return True
            else:
                return False
        elif self.set_additional_goal == 'vel':
            if np.sqrt((x - self.goal_state[0]) ** 2 + (y - self.goal_state[1]) ** 2) <= self.goal_pos_tolerance \
                and abs(vel - self.goal_state[3]) < 0.25:
                logger.log("in goal with specific velocity!!")
                return True
            else:
                return False
        else:
            raise ValueError("invalid param for set_additional_goal!")

    def get_obsrv(self, laser_data, dynamic_data):

        discretized_laser_data = self._discretize_laser(laser_data, self.num_lasers)

        # here dynamic_data is specially for 'mobile_based' since I specified model name
        x = dynamic_data.pose.position.x  # absolute x position
        y = dynamic_data.pose.position.y  # absolute y position
        ox = dynamic_data.pose.orientation.x
        oy = dynamic_data.pose.orientation.y
        oz = dynamic_data.pose.orientation.z
        ow = dynamic_data.pose.orientation.w
        _, _, theta = euler_from_quaternion([ox, oy, oz, ow])  # heading angle, which == yaw

        v = dynamic_data.twist.linear.x
        w = dynamic_data.twist.angular.z

        obsrv = [x, y, theta] + discretized_laser_data + [v, w]

        return obsrv

    def reset(self, spec=None):
        self.laser_data = LaserScan()
        self.contact_data = ContactsState()

        # Resets the simulation
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_world()
        except rospy.ServiceException as e:
            print("# Reset simulation fails!")

        # Set robot to random starting point
        if spec is not None:
            pose = Pose()
            pose.position.x = spec[0]
            pose.position.y = spec[1]
            pose.position.z = self.get_model_state(model_name="mobile_base").pose.position.z
            theta = spec[2]
            ox, oy, oz, ow = quaternion_from_euler(0.0, 0.0, theta)
            pose.orientation.x = ox
            pose.orientation.y = oy
            pose.orientation.z = oz
            pose.orientation.w = ow
        else:
            pose = Pose()
            pose.position.x = np.random.uniform(low=START_STATE[0] - 0.5, high=START_STATE[0] + 0.5)
            pose.position.y = np.random.uniform(low=START_STATE[1] - 0.5, high=START_STATE[1] + 0.5)
            pose.position.z = self.get_model_state(model_name="mobile_base").pose.position.z
            theta = np.random.uniform(low=START_STATE[2], high=START_STATE[2] + np.pi)
            ox, oy, oz, ow = quaternion_from_euler(0.0, 0.0, theta)
            pose.orientation.x = ox
            pose.orientation.y = oy
            pose.orientation.z = oz
            pose.orientation.w = ow

        reset_state = ModelState()
        reset_state.model_name = "mobile_base"
        reset_state.pose = pose
        self.set_model_state(reset_state)

        # Prepare receive sensor readings
        laser_data = self.get_laser()
        new_laser_data = laser_data

        # Unpause simulation only for obtaining observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print ("/gazebo/unpause_physics service call failed")

        while new_laser_data.header.stamp <= laser_data.header.stamp:
            new_laser_data = self.get_laser()

        # Pause simulator to do other operations
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print ("/gazebo/pause_physics service call failed")

        # read dynamics data
        dynamic_data = None
        rospy.wait_for_service("/gazebo/get_model_state")
        while dynamic_data is None:
            dynamic_data = self.get_model_state(model_name="mobile_base")

        obsrv = self.get_obsrv(new_laser_data, dynamic_data)
        self.pre_obsrv = obsrv
        self.pre_vel = obsrv[-2]     # linear_vel
        self.pre_angvel = obsrv[-1]  # ang_vel

        return np.asarray(obsrv)

    def step(self, action):

        # Check for possible nan action
        if sum(np.isnan(action)) > 0:
            raise ValueError("Passed in nan to step! Action: " + str(action))

        action = np.clip(action, -2, 2)

        # For linear acc, [-2, 2] --> [-0.5, 0.5]
        linear_acc = -0.5 + (0.5 - (-0.5)) * (action[0] - (-2)) / (2 - (-2))
        # For angular acc, [-2, 2] --> [-0.1, 0.1]
        angular_acc = -0.1 + (0.1 - (-0.1)) * (action[1] - (-2)) / (2 - (-2))



        # Publish control command
        vel_cmd = Twist()
        vel_cmd.linear.x  = self.pre_vel + linear_acc
        vel_cmd.angular.z = self.pre_angvel + angular_acc


        print("before send cmd, linear_acc: {}; angular_acc: {}; pre_vel: {}, pre_angvel: {}, linear_vel: {}; angular_vel: {}".format(linear_acc, angular_acc, self.pre_vel, self.pre_angvel, vel_cmd.linear.x, vel_cmd.angular.z))
        self.vel_pub.publish(vel_cmd)

        # Prepare for receive sensor readings. Laser data as part of obs; contact data used for collision detection
        contact_data = self.get_contact()
        laser_data   = self.get_laser()
        new_contact_data = contact_data
        new_laser_data   = laser_data

        # Unpause simulation only for obtaining valid data streaming
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        while new_contact_data.header.stamp <= contact_data.header.stamp or \
                new_laser_data.header.stamp <= laser_data.header.stamp:
            new_contact_data = self.get_contact()
            new_laser_data   = self.get_laser()

        # Pause the simulation to do other operations
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        # Call a service to get model state
        dynamic_data = None
        rospy.wait_for_service("/gazebo/get_model_state")
        while dynamic_data is None:
            dynamic_data = self.get_model_state(model_name="mobile_base")

        obsrv = self.get_obsrv(new_laser_data, dynamic_data)

        # special solution for nan/inf observation (especially in case of any invalid sensor readings)
        if any(np.isnan(np.array(obsrv))) or any(np.isinf(np.array(obsrv))):
            logger.record_tabular("found nan or inf in observation:", obsrv)
            obsrv = self.pre_obsrv
            done = True
            self.step_counter = 0

        self.pre_obsrv = obsrv
        self.pre_vel   = obsrv[-2]
        self.pre_angvel= obsrv[-1]

        assert self.reward_type is not None
        reward = 0

        if self.reward_type == 'hand_craft':
            reward += 0
        else:
            raise ValueError("reward type is invalid!")

        done = False
        suc  = False
        self.step_counter += 1
        event_flag = None  # {'collision', 'safe', 'goal', 'steps exceeding'}

        # 1. Check collision. If something is wrong, go check old code to specify another _in_obst function
        if self._in_obst(new_contact_data):
            reward += self.collision_reward
            done = True
            self.step_counter = 0
            event_flag = 'collision'

        # 2. In the neighbor of goal state, done is True as well. Only considering velocity and pos
        if self._in_goal(np.array(obsrv[:3] + obsrv[-2:])):
            reward += self.goal_reward
            done = True
            suc  = True
            self.step_counter = 0
            event_flag = 'goal'

        # 3. If reaching maximum episode step, then we reset and give penalty.
        if self.step_counter >= 300:
            reward += self.collision_reward
            done = True
            self.step_counter = 0
            event_flag = 'steps exceeding'

        if event_flag is None:
            event_flag = 'safe'

        return np.asarray(obsrv), reward, done, {'suc':suc, 'event':event_flag}


# small test code
if __name__ == "__main__":
    env = DubinsCarEnv_v0(reward_type='hand_craft', set_additional_goal="angle")
    obs = env.reset()
    while True:
        obs, r, d, _ = env.step([0, 2.0])
        if d:
            obs = env.reset()