import numpy as np
from brs_engine.PlanarQuad_brs_engine import PlanarQuad_brs_engine
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

    def init_quad(self, srv, x, y, phi):
        quad_state = None

        quad_pose = Pose()
        quad_pose.position.x = x
        quad_pose.position.y = 0
        # here y(a wrong name here) --> z(Gazebo)
        quad_pose.position.z = y

        qu_x, qu_y, qu_z, qu_w = quaternion_from_euler(0, phi, 0)
        quad_pose.orientation.x = qu_x
        quad_pose.orientation.y = qu_y
        quad_pose.orientation.z = qu_z
        quad_pose.orientation.w = qu_w

        quad_state = ModelState()
        quad_state.model_name = "quadrotor"
        quad_state.pose = quad_pose

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
        # mod = int(len(laser_data.ranges)/new_ranges)
        # for i, item in enumerate(laser_data.ranges):
        #     if (i+1) % mod == 0:
        #         if laser_data.ranges[i] == float('Inf') or np.isinf(laser_data.ranges[i]):
        #             discretized_ranges.append(10)
        #         elif np.isnan(laser_data.ranges[i]):
        #             discretized_ranges.append(0)
        #         else:
        #             discretized_ranges.append(int(laser_data.ranges[i]))

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


    def gen_data(self, brs_engine, data_form='valueFunc', use='train'):
        if use == 'train':
            filepath = CUR_PATH + '/data/' + data_form + '_' + use + '.csv'
            # assert not os.path.exists(filepath)
            num = 35000
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = ['x', 'vx', 'y', 'vy', 'phi', 'w', 'value', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                gMin = np.asarray(brs_engine.gMin)
                gMax = np.asarray(brs_engine.gMax)
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

                        x   = np.random.uniform(low=gMin[0,0], high=gMax[0,0])
                        vx  = np.random.uniform(low=gMin[1,0], high=gMax[1,0])
                        # optCtrl problem y:[-5,5], but in Gazebo, it's [0,10]
                        y   = np.random.uniform(low=gMin[2,0], high=gMax[2,0])
                        vy  = np.random.uniform(low=gMin[3,0], high=gMax[3,0])
                        phi = np.random.uniform(low=gMin[4,0], high=gMax[4,0])
                        w   = np.random.uniform(low=gMin[5,0], high=gMax[5,0])
                        value = brs_engine.evaluate_value([x,vx,y,vy,phi,w])

                        print("initial state:", [x,vx,y,vy,phi,w])
                        self.init_quad(self.srv_set_model_state, x, y+5, phi)

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
                        tmp_dict = {'x': x, 'vx': vx, 'y':y, 'vy':vy, 'phi':phi, 'w':w, 'value':value[0], \
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


class SL_valueLearner(object):
    def __init__(self):
        self.batch_size  = 50
        self.input_size  = 6 + 8
        self.keep_prob   = 1
        self.epoch = 50
        self.train_data_size = 35000
        self.test_data_size  = 15000

    def read_data(self, use='train'):
        if use == 'train':
            data = pd.read_csv('./data/valueFunc_train.csv')
        elif use == 'test':
            data = pd.read_csv('./data/valueFunc_test.csv')
        length = len(data)

        X = []
        y = []

        for dx, values in data.iterrows():
            vect = []
            vect.extend(values[0:6])
            vect.extend(values[7:])
            # print("vect", vect)
            vect = np.array(vect)
            X.append(vect)

            v_gt = values[6]
            y.append([v_gt])

        X = np.array(X)
        y = np.array(y)

        print("read data for " + use + ' ...', np.shape(X))
        print("read label for " + use + ' ...', np.shape(y))

        return X,y


    def dense_layer(input, in_size, out_size, activation_function=None):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([out_size]) + 0.1)
        Wx_plus_b = tf.matmul(input, Weights) + biases  # not actived yet
        Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob=keep_prob_s)
        if activation_function is None:
            output = Wx_plus_b
        else:
            output = activation_function(Wx_plus_b)
        return output


    def build_graph(self):
        self.xs = tf.placeholder(tf.float32, [None, self.input_size])
        self.ys = tf.placeholder(tf.float32, [None, 1])

        # TODO: maybe need to normalize input xs
        # xs = xs.normalize()
        self.hidden_out1 = dense_layer(xs, self.input_size, 64, activation_function=tf.nn.tanh)
        self.hidden_out2 = dense_layer(hidden_out1, 64, 64, activation_function=tf.nn.tanh)
        self.vpred = dense_layer(hidden_out2, 64, 1, activation_function=None)

        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - vpred), reduction_indices=[1]))
        self.train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

    def train(self):
        X_train, y_train = read_data(use='train')

        with tf.Session() as sess:
            init = tf.initialize_all_variables()
            sess.run(init)

            # total iter_num = data_size / batch_size * epoch_num
            for iter in range(self.epoch * self.train_data_size / self.batch_size):
                start = (iter * batch_size) % self.train_data_size
                end   = min(start + batch_size, self.train_data_size)

                _, pred, loss = sess.run([train_step, vpred, loss], feed_dict={xs: X_train[start:end], ys: y_train[start:end], keep_prob_s: keep_prob})

                if iter % 50 == 0:
                    print("iter: ", '%04d' % (iter + 1), "loss: ", los)

    def test(self):
        X_test, y_test = read_data(use='test')

        # TODO: load trained model for test


if __name__ == "__main__":

    # data_gen = Data_Generator()
    # quad_brs_engine = PlanarQuad_brs_engine()
    # data_gen.gen_data(quad_brs_engine, data_form='valueFunc', use='train')

    vlnr = SL_valueLearner()
    vlnr.read_data()