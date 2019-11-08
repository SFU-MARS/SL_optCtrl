import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import gazebo_env
from utils.utils import *
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

from gazebo_msgs.msg import ContactsState

import rospy
import time
import copy

from baselines import logger

from tf.transformations import euler_from_quaternion, quaternion_from_euler

# FOR TEST NEW WORLD.
# need to be compatitable with model.sdf and world.sdf for custom setting

# notice: it's not the gazebo pose state, not --> x,y,z,pitch,roll,yaw !!
GOAL_STATE = np.array([4.0, 0., 9., 0., 0.75, 0.])

# START_STATE = np.array([3.18232, 0., 3., 0., 0., 0.])
# START_STATE = np.array([-1.5, -3.33973, 2.5, 0., 0., 0.])
START_STATE = np.array([3.5, 0, 2.5, 0, 0, 0])

# obstacles position groundtruth
# OBSTACLES_POS = [(-1.5, 3), (3, 6), (-2, 7)]
# (xpos, zpos, xsize, zsize)
OBSTACLES_POS = [(-2, 5, 1.5, 1.5),
                 (0, 8.5, 1.5, 1.0),
                 (3.5, 5, 3, 1.5)]

# wall 0,1,2,3
WALLS_POS = [(-5., 5.), (5., 5.), (0.0, 9.85), (0.0, 5.)]


class PlanarQuadEnv_v0(gazebo_env.GazeboEnv):
    def __init__(self, reward_type, set_additional_goal, **kwargs):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "QuadrotorAirSpace_v0.launch")

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.apply_joint_effort = rospy.ServiceProxy('/gazebo/apply_joint_effort', ApplyJointEffort)
        self.clear_joint_effort = rospy.ServiceProxy('/gazebo/clear_joint_forces', JointRequest)
        self.force = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)


        self._seed()

        self.m = 1.25
        self.g = 9.81
        self.num_lasers = 8
        self.Thrustmax = 0.75 * self.m * self.g
        self.Thrustmin = 0
        self.control_reward_coff = 0.01
        self.collision_reward = -2 * 200 * self.control_reward_coff * (self.Thrustmax ** 2)
        self.goal_reward = 1000

        self.start_state = START_STATE
        self.goal_state = GOAL_STATE

        # state space and action space (MlpPolicy needs these params for input)
        high_state = np.array([5., 2., 10., 2., np.pi, np.pi / 3])
        low_state = np.array([-5., -2., 0., -2., -np.pi, -np.pi / 3])

        high_obsrv = np.array([5., 2., 10., 2., np.pi, np.pi / 3] + [5*2] * self.num_lasers)
        low_obsrv = np.array([-5., -2., 0., -2., -np.pi, -np.pi/3] + [0] * self.num_lasers)

        # controls are two thrusts
        # here high_action and low_action is only used by DDPG and stable_baseline's PPO2. Set [-1,1] is to be consistent with the default DDPG action range
        # high_action = np.array([1., 1.])
        # low_action  = np.array([-1., -1.])
        high_action = np.array([12.0, 12.0])
        low_action  = np.array([0.0, 0.0])

        self.state_space = spaces.Box(low=low_state, high=high_state)
        self.observation_space = spaces.Box(low=low_obsrv, high=high_obsrv)
        self.action_space = spaces.Box(low=low_action, high=high_action)

        self.state_dim = 6
        self.action_dim = 2

        self.goal_pos_tolerance = 1.0
        # self.goal_pos_tolerance = 0.5
        self.goal_vel_limit = 0.25
        self.goal_phi_limit = np.pi / 6.


        self.pre_obsrv = None
        self.reward_type = reward_type
        self.set_additional_goal = set_additional_goal

        # no need any more
        # self.brsEngine = None

        # used to monitor episode steps
        self.step_counter = 0

        # make it consistent between MPC and RL control stepsize. MPC: 0.05s; RL: 0.05s as well set at .world file
        self.sim_stepcounter = 0
        self.old_wrench = None
        self.MPC_RL_factor = 5

        # used for customized reset function. Type: list of reset state (6D) or None
        self.customized_reset = None
        self.num_envs = 1

    def _discretize_laser(self, laser_data, new_ranges):

        discretized_ranges = []

        full_ranges = float(len(laser_data.ranges))
        # print("laser ranges num: %d" % full_ranges)

        for i in range(new_ranges):
            new_i = int(i * full_ranges // new_ranges + full_ranges // (2 * new_ranges))
            if laser_data.ranges[new_i] == float('Inf') or np.isinf(laser_data.ranges[new_i]):
                # discretized_ranges.append(float('Inf'))
                discretized_ranges.append(10)
            elif np.isnan(laser_data.ranges[new_i]):
                discretized_ranges.append(float('Nan'))
                # discretized_ranges.append(0)
            else:
                discretized_ranges.append(laser_data.ranges[new_i])
                # discretized_ranges.append(int(laser_data.ranges[new_i]))

        return discretized_ranges

    """
    def _in_obst(self, contact_data):

        if len(contact_data.states) != 0:
            if contact_data.states[0].collision1_name != "" and contact_data.states[0].collision2_name != "":
                return True
        else:
            return False
    """

    def _in_obst(self, laser_data, dynamic_data):
         laser_min_range = 0.6
         # collision_min_range = 0.8
         tmp_x = dynamic_data.pose.position.x
         tmp_y = dynamic_data.pose.position.y
         tmp_z = dynamic_data.pose.position.z

         if tmp_z <= laser_min_range:
             # print("tmp_z:", tmp_z)
             return True
         if Euclid_dis((tmp_x, tmp_z), (OBSTACLES_POS[0][0], OBSTACLES_POS[0][1])) < 0.5*np.sqrt((OBSTACLES_POS[0][2])**2 + (OBSTACLES_POS[0][3])**2) + 0.5 \
            or Euclid_dis((tmp_x, tmp_z), (OBSTACLES_POS[1][0], OBSTACLES_POS[1][1])) < 0.5*np.sqrt((OBSTACLES_POS[1][2])**2 + (OBSTACLES_POS[1][3])**2) + 0.5 \
            or Euclid_dis((tmp_x, tmp_z), (OBSTACLES_POS[2][0], OBSTACLES_POS[2][1])) < 0.5*np.sqrt((OBSTACLES_POS[2][2])**2 + (OBSTACLES_POS[2][3])**2) + 0.5 \
            or np.abs(tmp_x - WALLS_POS[0][0]) < laser_min_range \
            or np.abs(tmp_x - WALLS_POS[1][0]) < laser_min_range \
            or np.abs(tmp_z - WALLS_POS[2][1]) < laser_min_range:
             return True

         # check the obstacle by sensor reflecting data
         # for idx, item in enumerate(laser_data.ranges):
         #     if laser_min_range > laser_data.ranges[idx] > 0:
         #         return True
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
                    and (abs(phi - self.goal_state[4]) < 0.30):
                print("in goal!!")
                return True
            else:
                return False

        elif self.set_additional_goal == 'vel':
            if np.sqrt((x - self.goal_state[0]) ** 2 + (z - self.goal_state[2]) ** 2) <= self.goal_pos_tolerance \
                    and abs(vx - self.goal_state[1]) < self.goal_vel_limit and abs(vz - self.goal_state[3]) < self.goal_vel_limit:
                print("in goal!!")
                return True
            else:
                return False

        else:
            raise ValueError("invalid param for set_additional_goal!")

    def _in_half_goal(self, state):
        assert len(state) == self.state_dim

        x = state[0]
        z = state[2]
        # print("z pos:", z)

        phi = state[4]
        # print("phi:", phi)

        vx = state[1]
        vz = state[3]

        if self.set_additional_goal == 'angle' or self.set_additional_goal == 'vel':
            if np.sqrt((x - self.goal_state[0]) ** 2 + (z - self.goal_state[2]) ** 2) <= self.goal_pos_tolerance:
                print("in half goal, pos reached!")
                return True
            else:
                return False
        else:
            raise ValueErrror('None additional goal does not have half_goal!!')

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
        # print("action:", action)
        # print("action shape:", action.shape)
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


        # if self.sim_stepcounter % self.MPC_RL_factor == 0:
        #     self.old_wrench = wrench
        # if self.sim_stepcounter == self.MPC_RL_factor:
        #     self.sim_stepcounter = 0
        # self.sim_stepcounter += 1
        #
        # rospy.wait_for_service('/gazebo/apply_body_wrench')
        # self.force(body_name="base_link", reference_frame="world", wrench=self.old_wrench, start_time=rospy.Time().now(), duration=rospy.Duration(1))

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


        laser_data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
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
            # reward = -self.control_reward_coff * (action[0] ** 2 + action[1] ** 2)
            # reward = -1
            reward += 0
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
            # print("in goal")

        # if abs(obsrv[4] - self.goal_state[4]) < 0.40:
        #     print("good tilting!")

        if obsrv[4] > 1.2 or obsrv[4] < -1.2:
            reward += self.collision_reward * 2
            done = True
            self.step_counter = 0
            # print("tilt too much")
        # maximum episode length allowed
        if self.step_counter >= 30:
            done = True
            self.step_counter = 0
            # print('exceed max length')

        if high_dim_ac_form:
            # for PPO2 Vectorized Env
            return np.asarray(obsrv), np.asarray([reward]), np.asarray([done]), [{'suc':suc}]
        else:
            return np.asarray(obsrv), reward, done, {'suc':suc}

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

if __name__ == "__main__":
    quadEnv = PlanarQuadEnv_v0()
    quadEnv.reward_type = "hand_craft"
    quadEnv.set_additional_goal = 'None'
    quadEnv.customized_reset = [-1.1254724508, -0.0746026367, 2.2481865046, 0.0826751712,  0.0529436985, 0]
    obs = quadEnv.reset()
    res = quadEnv.step([5.0006609471, 10.4238665043])
    print("reset obs:", obs)
    print("next obs:", res[0])






