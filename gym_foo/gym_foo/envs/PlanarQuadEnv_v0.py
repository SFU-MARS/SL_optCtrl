import gym
from gym import spaces
from gym.utils import seeding
from utils.tools import *
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist, Pose
from geometry_msgs.msg import Wrench

from sensor_msgs.msg import LaserScan

from std_srvs.srv import Empty
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import ApplyJointEffort
from gazebo_msgs.srv import JointRequest # the type of clear joint effort
from gazebo_msgs.srv import ApplyBodyWrench

import rospy
import copy

from utils import logger

from tf.transformations import euler_from_quaternion, quaternion_from_euler

# FOR TEST NEW WORLD.
# need to be compatitable with model.sdf and world.sdf for custom setting


GOAL_ANGLE_RANGE = [0, np.pi/3]
# GOAL_ANGLE_RANGE = [-np.pi/4, 0]
GOAL_ANGLE_CENTER = GOAL_ANGLE_RANGE[0] + abs(GOAL_ANGLE_RANGE[1]-GOAL_ANGLE_RANGE[0])/2
GOAL_ANGLE_RADIUS = abs(GOAL_ANGLE_RANGE[1]-GOAL_ANGLE_RANGE[0])/2
logger.log("goal angle range: from {} to {}".format(GOAL_ANGLE_RANGE[0] * 180 / np.pi, GOAL_ANGLE_RANGE[1] * 180 / np.pi))
logger.log("goal angle center: {}".format(GOAL_ANGLE_CENTER * 180 / np.pi))
logger.log("goal angle radius: {}".format(GOAL_ANGLE_RADIUS * 180 / np.pi))

# notice: it's not the gazebo pose state, not --> x,y,z,pitch,roll,yaw !!
# GOAL_STATE = np.array([4.0, 0., 9., 0., 0.75, 0.])
# GOAL_STATE = np.array([4.0, 0, 9.0, 0, -np.pi/6, 0])
# GOAL_STATE = np.array([4.0, 0, 9.0, 0, 0, 0])
# GOAL_STATE = np.array([4.0, 0, 9.0, 0, -np.pi/8, 0])
# GOAL_STATE = np.array([4.0, 0, 9.0, 0, np.pi/6, 0])
GOAL_STATE = np.array([4.0, 0.0, 9.0, 0.0, GOAL_ANGLE_CENTER, 0.0])

# GOAL_PHI_LIMIT = np.pi/8
# GOAL_PHI_LIMIT = 0.3
# GOAL_PHI_LIMIT = np.pi/6
GOAL_PHI_LIMIT = GOAL_ANGLE_RADIUS



START_STATE = np.array([3.75, 0, 2, 0, 0, 0])
# START_STATE = np.array([3.5, 0, 2.5, 0, 0, 0])
# START_STATE = np.array([-2.5, 0, 2.5, 0, 0, 0])
# START_STATE = np.array([0, 0, 2, 0, 0, 0])


# obstacles position groundtruth
# (xpos, zpos, xsize, zsize)
# air_space_201910
# OBSTACLES_POS = [(-2, 5, 1.5, 1.5),
#                  (0, 8.5, 1.5, 1.0),
#                  (3.5, 5, 3, 1.5)]
# air_space_202002; in this env, baseline and mpc_init all can not handle it.
# OBSTACLES_POS = [(-3.5, 6, 3/2, 1/2),
#                  (0, 9, 1.5/2, 2/2),
#                  (3.5+0.3+0.2, 6, 3/2, 1/2),
#                  (-1, 4+0.2, 3/2, 0.5/2)]
# air_space_202002_Francis
OBSTACLES_POS = [(-2, 5, 1.5/2, 1.5/2),
                 (1, 8.5, 1.5/2, 1.0/2),
                 (3.5+0.25, 5, 3/2, 1.5/2),
                 (0, 1, 0.5/2, 2/2)]

# wall 0,1,2,3
WALLS_POS = [(-5., 5.), (5., 5.), (0.0, 9.85), (0.0, 5.0)]


class PlanarQuadEnv_v0(gym.Env):
    def __init__(self, reward_type, set_additional_goal, **kwargs):
        self.port = "11311"
        self.port_gazebo = "11345"
        os.environ["ROS_MASTER_URI"] = "http://localhost:" + self.port
        os.environ["GAZEBO_MASTER_URI"] = "http://localhost:" + self.port_gazebo
        rospy.init_node('PlanarQuadEnv', anonymous=True)

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.apply_joint_effort = rospy.ServiceProxy('/gazebo/apply_joint_effort', ApplyJointEffort)
        self.clear_joint_effort = rospy.ServiceProxy('/gazebo/clear_joint_forces', JointRequest)
        self.force = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)

        # cancel additional seed, because we already have it in train_ppo.py
        # self._seed()

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

        # make it consistent between MPC and RL control stepsize. MPC: 0.05s; RL: 0.05s as well set at .world file
        self.sim_stepcounter = 0
        self.old_wrench = None
        self.MPC_RL_factor = 5

        # used for customized reset function. Type: list of reset state (6D) or None
        self.customized_reset = None
        self.num_envs = 1

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


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


    def _in_obst(self, laser_data, dynamic_data):
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
        # print("z pos:", z)

        phi = state[4]
        # print("phi:", phi)

        vx = state[1]
        vz = state[3]

        # just consider pose restriction
        if self.set_additional_goal == 'None':
            if np.sqrt((x - self.goal_state[0]) ** 2 + (z - self.goal_state[2]) ** 2) <= self.goal_pos_tolerance:
                print("in goal!!")
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
                print("in goal with special velocity!!")
                return True
            else:
                return False

        else:
            raise ValueError("invalid param for set_additional_goal!")


    def get_obsrv(self, laser_data, dynamic_data):

        discretized_laser_data = self._discretize_laser(laser_data, 8)

        # planar quadrotor x position
        x = dynamic_data.pose.position.x
        # planar quadrotor z position
        z = dynamic_data.pose.position.z

        # planar quadrotor velocity at x axis,
        vx = dynamic_data.twist.linear.x
        # planar quadrotor velocity at y axis == real world velocity z axis
        vz = dynamic_data.twist.linear.z

        ox = dynamic_data.pose.orientation.x
        oy = dynamic_data.pose.orientation.y
        oz = dynamic_data.pose.orientation.z
        ow = dynamic_data.pose.orientation.w

        # planar quadrotor pitch angle (along x-axis)
        _, pitch, _ = euler_from_quaternion([ox, oy, oz, ow])

        # planar quadrotor pitch angular velocity
        w = dynamic_data.twist.angular.y

        obsrv = [x, vx, z, vz, pitch, w] + discretized_laser_data

        # print("obs:", obsrv)
        return obsrv

    def reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            if self.customized_reset is None:
                self.reset_proxy()
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

                reset_state = ModelState()
                reset_state.model_name = "quadrotor"
                reset_state.pose = pose
                self.set_model_state(reset_state)
                dynamic_data = self.get_model_state(model_name="quadrotor")
            else:
                # print("proceeding customized resetting function!!")
                cus_reset = self.customized_reset[np.random.choice(len(self.customized_reset))]
                self.reset_proxy()
                pose = Pose()
                pose.position.x = cus_reset[0]
                pose.position.y = self.get_model_state(model_name="quadrotor").pose.position.y
                pose.position.z = cus_reset[2]
                pitch = cus_reset[4]
                ox, oy, oz, ow = quaternion_from_euler(0.0, pitch, 0.0)
                pose.orientation.x = ox
                pose.orientation.y = oy
                pose.orientation.z = oz
                pose.orientation.w = ow

                twist = Twist()
                twist.linear.x = cus_reset[1]
                twist.linear.y = 0
                twist.linear.z = cus_reset[3]
                twist.angular.y = cus_reset[5]
                reset_state = ModelState()
                reset_state.model_name = "quadrotor"
                reset_state.pose = pose
                reset_state.twist = twist
                self.set_model_state(reset_state)
                dynamic_data = self.get_model_state(model_name="quadrotor")
                # print("dynamics data:", dynamic_data)
        except rospy.ServiceException as e:
            print("# Resets the state of the environment and returns an initial observation.")

        # rospy.sleep(5.)

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        # read laser data
        laser_data = rospy.wait_for_message('/scan', LaserScan, timeout=5)

        # pause to return obsrv
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        obsrv = self.get_obsrv(laser_data, dynamic_data)
        self.pre_obsrv = obsrv

        # print("obsrv's shape:", np.shape(np.asarray(obsrv)))
        return np.asarray(obsrv)

    def step(self, action):
        if len(np.shape(action)) > 1:
            high_dim_ac_form = True
            action = np.squeeze(action)
        else:
            high_dim_ac_form = False
        if sum(np.isnan(action)) > 0:
            raise ValueError("Passed in nan to step! Action: " + str(action))

        # --- previously direct shift mean of Gaussian from 0 to 8.8 around ---
        # print("action:",action)
        # action = action + 8.8  # a little more power for easier launch away from ground
        # --------------------------------------------------------------------

        # --- now, try action transformation [-2,2] -> [7,10] (for PPO and TRPO, because we are not using strict action range for them) ---
        # action = [7 + (10 - 7) * (a_i - (-2)) / (2 - (-2)) for a_i in action]
        # --------------------------------------------------------------------


        # --- no matter if we use pol_load. Remember when doing supervised learning, normalize obs and rescale actions ---
        # --- use [-1,1] is to be consisten with DDPG and PPO2 (from stable_baselines) default action range ---
        # --- rescale from [-1,1] -> [0,12] because MPC control range is [0,12], not [7,10] ---

        # print("action before:", action)
        action = np.clip(action, -2, 2)
        # action = 2.0 * np.tanh(action)
        # action = [0 + (12 - 0) * (a_i - (-1)) / (1- (-1)) for a_i in action]
        action = [7 + (10 - 7) * (a_i - (-2)) / (2 - (-2)) for a_i in action]
        # print("action after:", action)


        pre_phi = self.pre_obsrv[4]
        wrench = Wrench()
        wrench.force.x = (action[0] + action[1]) * np.sin(pre_phi)
        wrench.force.y = 0
        # wrench.force.z = action[0] + action[1]
        wrench.force.z = (action[0] + action[1]) * np.cos(pre_phi)
        wrench.torque.x = 0
        # wrench.torque.y = (action[0] - action[1]) * 0.5
        wrench.torque.y = (action[0] - action[1]) * 0.4
        # wrench.torque.y = 1.0
        wrench.torque.z = 0


        rospy.wait_for_service('/gazebo/apply_body_wrench')
        self.force(body_name="base_link", reference_frame="world", wrench=wrench, start_time=rospy.Time().now(), duration=rospy.Duration(1))
        # self.force(body_name="base_link", reference_frame="base_link", wrench=wrench, start_time=rospy.Time().now(), duration=rospy.Duration(1))

        dynamic_data = self.get_model_state(model_name="quadrotor")
        # print("dynamics data after one step:", dynamic_data)

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")


        laser_data = rospy.wait_for_message('/scan', LaserScan, timeout=20)
        # contact_data = rospy.wait_for_message('/gazebo_ros_bumper', ContactsState, timeout=50)
        # print("contact data:", contact_data)

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        done = False
        suc = False
        self.step_counter += 1
        event_flag = None  # {'collision', 'safe', 'goal', 'steps exceeding', 'highly tilt'}

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
            reward += 0
        elif self.reward_type == 'hand_craft_mpc':
            # reward = -self.control_reward_coff * (action[0] ** 2 + action[1] ** 2)
            # reward = -1
            # reward += 0

            # print("using hand_craft_mpc")
            delta_x = obsrv[0] - GOAL_STATE[0]
            delta_z = obsrv[2] - GOAL_STATE[2]
            delta_theta = obsrv[4] - GOAL_STATE[4]

            reward += -1.0 * (action[0] ** 2 + action[1] ** 2)
            reward += -10000.0 * (delta_x ** 2 + delta_z ** 2)
            reward = reward * 0.0001

            # print("delta x: {}".format(delta_x), "delta z: {}".format(delta_z), "reward from control: {}".format(-1.0 * (action[0] ** 2 + action[1] ** 2)),
            #       "reward from state diff: {}".format(-100 * (delta_x ** 2 + delta_z ** 2)))
        elif self.reward_type == "hand_craft_mpc_without_control":
            delta_x = obsrv[0] - GOAL_STATE[0]
            delta_z = obsrv[2] - GOAL_STATE[2]
            delta_theta = obsrv[4] - GOAL_STATE[4]

            reward += -np.sqrt(delta_x ** 2 + delta_z ** 2 + 10.0 * delta_theta ** 2)
        elif self.reward_type == "hand_craft_mpc_without_control_2":
            delta_x = obsrv[0] - GOAL_STATE[0]
            delta_z = obsrv[2] - GOAL_STATE[2]

            reward += -np.sqrt(delta_x ** 2 + delta_z ** 2)

        elif self.reward_type == 'ttr' and self.brsEngine is not None:
            # Notice z-axis ttr space is defined from (-5,5), in gazebo it's in (0,10), so you need -5 when you want correct ttr reward
            ttr_obsrv = copy.deepcopy(obsrv)
            # because in brs_engine, z pos is defined as [-5,5]. But here, z pos is defined as [0,10]
            ttr_obsrv[2] = ttr_obsrv[2] - 5
            ttr = self.brsEngine.evaluate_ttr(np.reshape(ttr_obsrv[:6], (1, -1)))
            reward += -ttr
        elif self.reward_type == 'distance':
            reward += -(Euclid_dis((obsrv[0], obsrv[2]), (GOAL_STATE[0], GOAL_STATE[2])))
            # reward += (-Euclid_dis((obsrv[0], obsrv[2]), (GOAL_STATE[0], GOAL_STATE[2])) - abs(obsrv[1]-GOAL_STATE[1]) - abs(obsrv[3]-GOAL_STATE[3]))
        elif self.reward_type == 'distance_lambda_0.1':
            delta_x = obsrv[0] - GOAL_STATE[0]
            delta_z = obsrv[2] - GOAL_STATE[2]
            delta_theta = obsrv[4] - GOAL_STATE[4]

            reward += -np.sqrt(delta_x**2 + delta_z**2 + 0.1 * delta_theta ** 2)
        elif self.reward_type == 'distance_lambda_1':
            delta_x = obsrv[0] - GOAL_STATE[0]
            delta_z = obsrv[2] - GOAL_STATE[2]
            delta_theta = obsrv[4] - GOAL_STATE[4]

            reward += -np.sqrt(delta_x ** 2 + delta_z ** 2 + 1.0 * delta_theta ** 2)
        elif self.reward_type == 'distance_lambda_10':
            delta_x = obsrv[0] - GOAL_STATE[0]
            delta_z = obsrv[2] - GOAL_STATE[2]
            delta_theta = obsrv[4] - GOAL_STATE[4]

            reward += -np.sqrt(delta_x ** 2 + delta_z ** 2 + 10.0 * delta_theta ** 2)
        else:
            raise ValueError("no option for step reward!")

        # print("step reward:", reward)
        # print("self.reward_type:", self.reward_type)


        # 1. when collision happens, done = True
        if self._in_obst(laser_data, dynamic_data):
            reward += self.collision_reward
            done = True
            self.step_counter = 0
            event_flag = 'collision'

        """
        if self._in_obst(contact_data):
            reward += self.collision_reward
            done = True
            self.step_counter = 0
            # print("obstacle!")
        """
        # 2. In the neighbor of goal state, done is True as well. Only considering velocity and pos
        if self._in_goal(np.array(obsrv[:6])):
            reward += self.goal_reward
            done = True
            suc = True
            self.step_counter = 0
            event_flag = 'goal'
            # print("in goal")

        # if abs(obsrv[4] - self.goal_state[4]) < 0.40:
        #     print("good tilting!")


        # Amend: modified by xlv, abs(obsrv[4]) > 1.2 -> abs(obsrv[4]) > 1.4
        if obsrv[4] > 1.4 or obsrv[4] < -1.4:
            reward += self.collision_reward * 2
            done = True
            self.step_counter = 0
            event_flag = 'highly tilt'
            # print("tilt too much")
        # maximum episode length allowed
        if self.step_counter >= 100:
            done = True
            self.step_counter = 0
            event_flag = 'steps exceeding'
            # print('exceed max length')

        if event_flag is None:
            event_flag = 'safe'

        if high_dim_ac_form:
            # for PPO2 Vectorized Env
            return np.asarray(obsrv), np.asarray([reward]), np.asarray([done]), [{'suc':suc}]
        else:
            return np.asarray(obsrv), reward, done, {'suc':suc, 'event':event_flag}






if __name__ == "__main__":
    quadEnv = PlanarQuadEnv_v0(reward_type = "hand_craft", set_additional_goal = 'None')
    obs = quadEnv.reset()
    while True:
        print("enter while loop!")
        # rospy.sleep(0.1)
        print(quadEnv.step([8, 8]))






