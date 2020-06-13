import gym
from gym import spaces
import rospy
import copy


from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist, Pose
from geometry_msgs.msg import Wrench
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import ApplyBodyWrench
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from utils.tools import *
from utils import logger


# GOAL_ANGLE_RANGE = [0, np.pi/3]   # for both air_space_202002_Francis and air_space_201910_ddpg
GOAL_ANGLE_RANGE = [-np.pi/3, np.pi/3]
GOAL_ANGLE_CENTER = GOAL_ANGLE_RANGE[0] + abs(GOAL_ANGLE_RANGE[1]-GOAL_ANGLE_RANGE[0])/2
GOAL_ANGLE_RADIUS = abs(GOAL_ANGLE_RANGE[1]-GOAL_ANGLE_RANGE[0])/2
logger.log("goal angle range: from {} to {}".format(GOAL_ANGLE_RANGE[0] * 180 / np.pi, GOAL_ANGLE_RANGE[1] * 180 / np.pi))
logger.log("goal angle center: {}".format(GOAL_ANGLE_CENTER * 180 / np.pi))
logger.log("goal angle radius: {}".format(GOAL_ANGLE_RADIUS * 180 / np.pi))

# goal and start state definitions
# START_STATE = np.array([3.75, 0, 2, 0, 0, 0])    # air_space_202002_Francis
# START_STATE = np.array([3, 0, 2, 0, 0, 0])       # air_space_201910_ddpg
START_STATE = np.array([2.75, 0, 2, 0, 0,0])       # test_for_Francis
GOAL_STATE = np.array([4.0, 0.0, 9.0, 0.0, GOAL_ANGLE_CENTER, 0.0])
GOAL_PHI_LIMIT = GOAL_ANGLE_RADIUS


# obstacle definitions from air_space_202002_Francis
# OBSTACLES_POS = [(-2, 5, 1.5/2, 1.5/2),
#                  (1, 8.5, 1.5/2, 1.0/2),
#                  (3.5+0.25, 5, 3/2, 1.5/2),
#                  (0, 1, 0.5/2, 2/2)]

# obstacle definitions from air_space_201910_ddpg (A simpler env designed for DDPG comparison)
# OBSTACLES_POS = [(-2, 5, 1.5/2, 1.5/2),
#                  (0, 8.5, 1.5/2, 1.0/2),
#                  (3.5+0.25, 5, 3/2, 1.5/2),
#                  (0, 1, 0.5/2, 2/2)]

# obstacle definitions from test_for_Francis
OBSTACLES_POS = [(-2, 5, 1.5/2, 1.5/2),
                 (1, 8.5, 1.5/2, 1.0/2),
                 (0, 1, 0.5/2, 2/2)]

# wall 0,1,2,3
WALLS_POS = [(-5., 5.), (5., 5.), (0.0, 9.85), (0.0, 5.0)]

logger.log("I'm running PlanarQuadEnv with subscriber ...")
class PlanarQuadEnv_v0(gym.Env):
    def __init__(self, reward_type='hand_craft', set_additional_goal='None'):
        # self.port = "11311"
        # self.port_gazebo = "11345"

        # self.port = "11411"
        # self.port_gazebo = "11445"
        # os.environ["ROS_MASTER_URI"] = "http://localhost:" + self.port
        # os.environ["GAZEBO_MASTER_URI"] = "http://localhost:" + self.port_gazebo
        rospy.init_node('PlanarQuadEnv', anonymous=True)

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.apply_force = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)

        # Define ros subscriber for receiving sensor readings. Note: this ways is more stable than using "wait_for_message"
        rospy.Subscriber("/scan", LaserScan, self._laser_scan_callback)
        self.laser_data = LaserScan()

        self.m = 1.25
        self.g = 9.81
        self.num_lasers = 8
        self.Thrustmax = 0.75 * self.m * self.g
        self.Thrustmin = 0
        self.control_reward_coff = 0.01
        self.collision_reward = -400
        self.goal_reward = 1000

        self.start_state = START_STATE
        self.goal_state = GOAL_STATE

        # state space and action space (MlpPolicy needs these params for input)
        high_state = np.array([5., 2., 10., 2., np.pi, np.pi/3])
        low_state = np.array([-5., -2., 0., -2., -np.pi, -np.pi/3])

        high_obsrv = np.array([5., 2., 10., 2., np.pi, np.pi/3] + [5*2] * self.num_lasers)
        low_obsrv = np.array([-5., -2., 0., -2., -np.pi, -np.pi/3] + [0] * self.num_lasers)

        # This is consistent with the step() function clipping range.
        # Since you cannot expect NN output is in line with your real physics property. So there always needs transformation
        high_action = np.array([2.0, 2.0])
        low_action = np.array([-2.0, -2.0])

        self.state_space = spaces.Box(low=low_state, high=high_state)
        self.observation_space = spaces.Box(low=low_obsrv, high=high_obsrv)
        self.action_space = spaces.Box(low=low_action, high=high_action)

        self.state_dim = 6
        self.action_dim = 2

        self.goal_pos_tolerance = 1.0
        self.goal_vel_limit = 0.25
        self.goal_phi_limit = GOAL_PHI_LIMIT

        self.pre_obsrv = None
        self.reward_type = reward_type
        self.set_additional_goal = set_additional_goal

        # used to monitor episode steps
        self.step_counter = 0
        self._max_episode_steps = 100

    def _laser_scan_callback(self, laser_msg):
        self.laser_data = laser_msg

    def get_laser(self):
        return self.laser_data


    def _discretize_laser(self, laser_data, new_ranges):

        discretized_ranges = []

        full_ranges = float(len(laser_data.ranges))

        for i in range(new_ranges):
            new_i = int(i * full_ranges // new_ranges + full_ranges // (2 * new_ranges))
            if laser_data.ranges[new_i] == float('Inf') or np.isinf(laser_data.ranges[new_i]):
                # discretized_ranges.append(float('Inf'))
                discretized_ranges.append(31.0)  # max_range: 30m, use max_range+1
            elif np.isnan(laser_data.ranges[new_i]):
                # discretized_ranges.append(float('Nan'))
                discretized_ranges.append(0.0)
            else:
                discretized_ranges.append(laser_data.ranges[new_i])

        return discretized_ranges


    def _in_obst(self, dynamic_data):
        laser_min_range = 0.6
        tmp_x = dynamic_data.pose.position.x
        tmp_y = dynamic_data.pose.position.y
        tmp_z = dynamic_data.pose.position.z

        if tmp_z <= laser_min_range:
            return True

        quad_r = 0.55
        for i, obs in enumerate(OBSTACLES_POS):
            if obs[0] - obs[2] - quad_r <= tmp_x <= obs[0] + obs[2] + quad_r and \
                    obs[1] - obs[3] - quad_r <= tmp_z <= obs[1] + obs[3] + quad_r:
                return True

        if np.abs(tmp_x - WALLS_POS[0][0]) < laser_min_range \
            or np.abs(tmp_x - WALLS_POS[1][0]) < laser_min_range \
            or np.abs(tmp_z - WALLS_POS[2][1]) < laser_min_range:
            return True

        return False


    def _in_goal(self, state):

        assert len(state) == self.state_dim

        x = state[0]
        z = state[2]

        phi = state[4]

        vx = state[1]
        vz = state[3]

        # just consider pose restriction
        if self.set_additional_goal == 'None':
            if np.sqrt((x - self.goal_state[0]) ** 2 + (z - self.goal_state[2]) ** 2) <= self.goal_pos_tolerance:
                logger.log("in goal!!")
                return True
            else:
                return False
        elif self.set_additional_goal == 'angle':
            # angle region from 0.4 to 0.3. increase difficulty
            if np.sqrt((x - self.goal_state[0]) ** 2 + (z - self.goal_state[2]) ** 2) <= self.goal_pos_tolerance \
                    and (abs(phi - self.goal_state[4]) <= self.goal_phi_limit):  # 0.30 before
                logger.log("in goal with special angle!!")
                logger.log("x:{}, z:{}, phi:{}".format(x, z, phi))
                return True
            else:
                return False

        elif self.set_additional_goal == 'vel':
            if np.sqrt((x - self.goal_state[0]) ** 2 + (z - self.goal_state[2]) ** 2) <= self.goal_pos_tolerance \
                    and abs(vx - self.goal_state[1]) <= self.goal_vel_limit and abs(vz - self.goal_state[3]) <= self.goal_vel_limit:
                logger.log("in goal with special velocity!!")
                return True
            else:
                return False

        else:
            raise ValueError("invalid param for set_additional_goal!")


    def get_obsrv(self, laser_data, dynamic_data):

        discretized_laser_data = self._discretize_laser(laser_data, 8)

        x = dynamic_data.pose.position.x  # planar quadrotor x position
        z = dynamic_data.pose.position.z  # planar quadrotor z position

        vx = dynamic_data.twist.linear.x  # planar quadrotor velocity at x axis,
        vz = dynamic_data.twist.linear.z  # planar quadrotor velocity at z axis

        ox = dynamic_data.pose.orientation.x
        oy = dynamic_data.pose.orientation.y
        oz = dynamic_data.pose.orientation.z
        ow = dynamic_data.pose.orientation.w

        _, pitch, _ = euler_from_quaternion([ox, oy, oz, ow])  # planar quadrotor pitch angle (along x-axis)
        w = dynamic_data.twist.angular.y   # planar quadrotor pitch angular velocity

        obsrv = [x, vx, z, vz, pitch, w] + discretized_laser_data

        return obsrv

    def reset(self, spec=None):
        # Resets subscribed data
        self.laser_data = LaserScan()

        # Resets the simulated world
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset_world()
        except rospy.ServiceException as e:
            print("# Reset simulation fails!")

        if spec is None:
            pose = Pose()
            pose.position.x = np.random.uniform(low=self.start_state[0] - 0.5, high=self.start_state[0] + 0.5)
            pose.position.y = self.get_model_state(model_name="quadrotor").pose.position.y
            pose.position.z = np.random.uniform(low=self.start_state[2] - 0.5, high=self.start_state[2] + 0.5)
            pitch = np.random.uniform(low=self.start_state[4] - 0.1, high=self.start_state[4] + 0.1)
            ox, oy, oz, ow = quaternion_from_euler(0.0, pitch, 0.0)
            pose.orientation.x = ox
            pose.orientation.y = oy
            pose.orientation.z = oz
            pose.orientation.w = ow

            twist = Twist()
            twist.linear.x = 0
            twist.linear.y = 0
            twist.linear.z = 0
            twist.angular.y = 0

        else:
            pose = Pose()
            pose.position.x = spec[0]
            pose.position.y = self.get_model_state(model_name="quadrotor").pose.position.y
            pose.position.z = spec[2]
            pitch = spec[4]
            ox, oy, oz, ow = quaternion_from_euler(0.0, pitch, 0.0)
            pose.orientation.x = ox
            pose.orientation.y = oy
            pose.orientation.z = oz
            pose.orientation.w = ow

            twist = Twist()
            twist.linear.x = spec[1]
            twist.linear.y = 0
            twist.linear.z = spec[3]
            twist.angular.y = spec[5]

        reset_state = ModelState()
        reset_state.model_name = "quadrotor"
        reset_state.pose = pose
        reset_state.twist = twist
        self.set_model_state(reset_state)

        # Prepare receive sensor readings
        laser_data = self.get_laser()
        new_laser_data = laser_data

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        while new_laser_data.header.stamp <= laser_data.header.stamp:
            new_laser_data = self.get_laser()

        # pause to do other operations
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        # read dynamics data
        dynamic_data = None
        rospy.wait_for_service("/gazebo/get_model_state")
        while dynamic_data is None:
            dynamic_data = self.get_model_state(model_name="quadrotor")

        obsrv = self.get_obsrv(new_laser_data, dynamic_data)
        self.pre_obsrv = obsrv

        return np.asarray(obsrv)

    def step(self, action):
        # Check for possible nan action
        if sum(np.isnan(action)) > 0:
            raise ValueError("Passed in nan to step! Action: " + str(action))

        # do some action transoformation
        action = np.clip(action, -2, 2)
        action = [7 + (10 - 7) * (a_i - (-2)) / (2 - (-2)) for a_i in action]

        pre_phi = self.pre_obsrv[4]
        wrench = Wrench()
        wrench.force.x = (action[0] + action[1]) * np.sin(pre_phi)
        wrench.force.y = 0
        wrench.force.z = (action[0] + action[1]) * np.cos(pre_phi)

        wrench.torque.x = 0
        wrench.torque.y = (action[0] - action[1]) * 0.4
        wrench.torque.z = 0

        # apply wrench command
        rospy.wait_for_service('/gazebo/apply_body_wrench')
        self.apply_force(body_name="base_link", reference_frame="world", wrench=wrench, start_time=rospy.Time().now(), duration=rospy.Duration(1))

        # Prepare for receive sensor readings. Laser data as part of obs;
        laser_data = self.get_laser()
        new_laser_data = laser_data

        # unpause physics to receive valid data streaming from simulator
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")


        while new_laser_data.header.stamp <= laser_data.header.stamp:
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
            dynamic_data = self.get_model_state(model_name="quadrotor")

        done = False
        suc = False
        self.step_counter += 1
        event_flag = None  # {'collision', 'safe', 'goal', 'steps exceeding', 'highly tilt'}

        obsrv = self.get_obsrv(new_laser_data, dynamic_data)

        # special solution for nan/inf observation (especially in case of any invalid sensor readings)
        if any(np.isnan(np.array(obsrv))) or any(np.isinf(np.array(obsrv))):
            logger.log("found nan or inf in observation in step function:", obsrv)
            obsrv = self.pre_obsrv
            done = True
            self.step_counter = 0
        self.pre_obsrv = obsrv

        # configure step reward
        assert self.reward_type is not None
        reward = 0

        if self.reward_type == 'hand_craft':
            reward += 0
        else:
            raise ValueError("no option for step reward!")

        # 1. when collision happens, done = True
        if self._in_obst(dynamic_data):
            reward += self.collision_reward
            done = True
            self.step_counter = 0
            event_flag = 'collision'

        # 2. In the neighbor of goal state, done is True as well. Only considering velocity and pos
        if self._in_goal(np.array(obsrv[:6])):
            reward += self.goal_reward
            done = True
            suc = True
            self.step_counter = 0
            event_flag = 'goal'

        # 3. When higly tilt happens
        if obsrv[4] > 1.4 or obsrv[4] < -1.4:
            reward += self.collision_reward * 2
            done = True
            self.step_counter = 0
            event_flag = 'highly tilt'

        # 4. when exceeds maximum episode length
        if self.step_counter >= 100:
            done = True
            self.step_counter = 0
            event_flag = 'steps exceeding'

        if event_flag is None:
            event_flag = 'safe'

        return np.asarray(obsrv), reward, done, {'suc':suc, 'event':event_flag}




if __name__ == "__main__":
    quadEnv = PlanarQuadEnv_v0(reward_type="hand_craft", set_additional_goal='None')
    obs = quadEnv.reset()
    while True:
        print("enter while loop!")
        # rospy.sleep(0.1)
        print(quadEnv.step([8, 8]))






