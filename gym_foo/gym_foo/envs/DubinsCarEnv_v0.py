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
print(sys.path)
import tf
print(tf.__file__)
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# same as the goal and start area definition in "playground.world" file
# GOAL_STATE = np.array([3.41924, 3.6939, 0])


##### Xubo Old Setting #####
# GOAL_STATE = np.array([3.5, 3.5, np.pi*11/18]) # around 110 degrees as angle
# GOAL_STATE = np.array([3.5, 3.5, 0])
# GOAL_STATE = np.array([3.5, 3.5, np.pi*3/4])
# START_STATE = np.array([-0.222404, -3.27274, 0])
#############################

##### Env for mbi env1 #####
START_STATE = np.array([-4, -4, 0])
OBSTACLE_POS = np.array([(1.0, 2.0, -4.7, -0.7),
                        (-2.7, -1.7, 1.6, 5),
                        (-1.7, 2.5, 1.6, 2.6)]) # e.g. [x1, x2, y1, y2]
# GOAL_STATE = np.array([-0.9, -4, -0.75]) # e.g. [x, y, radius]
GOAL_STATE = np.array([0, 0, np.pi*2]) # e.g. [x, y, degree]

ENV_RANGE = np.array([-4.94, 4.66, -4.7745, 4.9]) # e.g. [x1, x2, y1, y2]
############################

"""
dubins' car 3d model:
x_dot = v * cos(theta)
y_dot = v * sin(theta)
theta_dot = w

"""
print("I'm running DubinsCarEnv with subscriber ...")
class DubinsCarEnv_v0(gym.Env):
    def __init__(self, reward_type="hand_craft", set_additional_goal="None"):
        # ROS and Gazebo environment variables setting
        # self.port = "11311"
        # self.port_gazebo = "11345"

        # self.port = "11411"
        # self.port_gazebo = "11445"

        # os.environ["ROS_MASTER_URI"] = "http://localhost:" + self.port
        # os.environ["GAZEBO_MASTER_URI"] = "http://localhost:" + self.port_gazebo
        # print(os.environ["ROS_MASTER_URI"])
        # print(os.environ["GAZEBO_MASTER_URI"])
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
        self.state_dim  = 3  # x,y,theta
        self.action_dim = 2  # v,w

        high_state = np.array([5.0, 5.0, np.pi])
        low_state  = np.array([-5.0, -5.0, -np.pi])

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
        self.goal_pos_tolerance = 0.75 # changed from 1 to 0.75 in mbi_env1
        self.goal_theta_tolerance = 0.75  # around 45 degrees

        self.control_reward_coff = 0.01
        self.collision_reward = -400
        self.goal_reward = 1000
        self.exceeding_reward = -800

        self.pre_obsrv = None
        self.reward_type = reward_type
        self.set_additional_goal = set_additional_goal

        self.step_counter = 0
        self._max_episode_steps = 200

        self.car_radius = 0.3
        logger.log("Gym dubins_car environment successfully initialized!!")

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
    # def _in_obst(self, contact_data):
    #     if len(contact_data.states) != 0:
    #         if contact_data.states[0].collision1_name != "" and contact_data.states[0].collision2_name != "":
    #             return True
    #     return False


    def _in_obst(self, state):
        x = state[0]
        y = state[1]
        for i, obs in enumerate(OBSTACLE_POS):
            if (obs[0] - self.car_radius <= x <= obs[1] + self.car_radius  and  
                obs[2] - self.car_radius <= y <= obs[3] + self.car_radius):
                return True
        if (ENV_RANGE[0] + self.car_radius <= x <= ENV_RANGE[1] - self.car_radius  and  
            ENV_RANGE[2] + self.car_radius <= y <= ENV_RANGE[3] - self.car_radius):
            return False
        else:
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

        obsrv = [x, y, theta] + discretized_laser_data


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
            pose.position.x = np.random.uniform(low=START_STATE[0] - 0.375, high=START_STATE[0] + 0.375)
            pose.position.y = np.random.uniform(low=START_STATE[1] - 0.375, high=START_STATE[1] + 0.375)
            pose.position.z = self.get_model_state(model_name="mobile_base").pose.position.z
            theta = np.random.uniform(low=START_STATE[2], high=START_STATE[2] + np.pi / 4)
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

        return np.asarray(obsrv)

    def step(self, action):

        # Check for possible nan action
        if sum(np.isnan(action)) > 0:
            raise ValueError("Passed in nan to step! Action: " + str(action))

        action = np.clip(action, -2, 2)

        # For linear vel, [-2, 2] --> [-0.8, 2]
        linear_vel = -0.8 + (2 - (-0.8)) * (action[0] - (-2)) / (2 - (-2))
        # For angular vel, [-2, 2] --> [-0.8, 0.8]. If something wrong happens, check old code to specify for PPO, DDPG or TRPO
        angular_vel = -0.8 + (0.8 - (-0.8)) * (action[1] - (-2)) / (2 - (-2))

        # Publish control command
        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = angular_vel
        self.vel_pub.publish(vel_cmd)

        # Prepare for receive sensor readings. Laser data as part of obs; contact data used for collision detection
        contact_data = self.get_contact()
        laser_data = self.get_laser()

        # new_contact_data = contact_data
        # new_laser_data   = laser_data

        # Unpause simulation only for obtaining valid data streaming
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        # while new_contact_data.header.stamp <= contact_data.header.stamp or \
        #         new_laser_data.header.stamp <= laser_data.header.stamp:
        #     new_contact_data = self.get_contact()
        #     new_laser_data   = self.get_laser()


        # Pause the simulation to do other operations
        # rospy.wait_for_service('/gazebo/pause_physics')
        # try:
        #     self.pause()
        # except rospy.ServiceException as e:
        #     print("/gazebo/pause_physics service call failed")

        # Call a service to get model state
        dynamic_data = None
        rospy.wait_for_service("/gazebo/get_model_state")
        while dynamic_data is None:
            dynamic_data = self.get_model_state(model_name="mobile_base")

        # obsrv = self.get_obsrv(new_laser_data, dynamic_data) # Xubo's old version
        obsrv = self.get_obsrv(laser_data, dynamic_data)


        # special solution for nan/inf observation (especially in case of any invalid sensor readings)
        if any(np.isnan(np.array(obsrv))) or any(np.isinf(np.array(obsrv))):
            logger.record_tabular("found nan or inf in observation:", obsrv)
            obsrv = self.pre_obsrv
            done = True
            self.step_counter = 0

        self.pre_obsrv = obsrv

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
        # if self._in_obst(new_contact_data):
        if self._in_obst(np.array(obsrv[:2])):
            reward += self.collision_reward
            done = True
            self.step_counter = 0
            event_flag = 'collision'

        # 2. In the neighbor of goal state, done is True as well. Only considering velocity and pos
        if self._in_goal(np.array(obsrv[:3])):
            reward += self.goal_reward
            done = True
            suc  = True
            self.step_counter = 0
            event_flag = 'goal'

        # 3. If reaching maximum episode step, then we reset and give penalty.
        if self.step_counter >= self._max_episode_steps:
            reward += self.exceeding_reward
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