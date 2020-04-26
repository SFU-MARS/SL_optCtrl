import sys,os
from keras.models import load_model
import csv
import pandas as pd
import numpy as np
import rospy
from geometry_msgs.msg import Twist, Pose
from geometry_msgs.msg import Wrench
from gazebo_msgs.msg import ModelState
from gazebo_msgs.msg import LinkState
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import GetLinkState
from gazebo_msgs.srv import SetLinkState
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.srv import ApplyBodyWrench
from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion, quaternion_from_euler

def norm(x, stats):
    return (x - stats['mean']) / stats['std']

def discretize_sensor(sensor_data, new_ranges):
    discretized_ranges = []
    full_ranges = float(len(sensor_data.ranges))
    # print("laser ranges num: %d" % full_ranges)

    for i in range(new_ranges):
        new_i = int(i * full_ranges // new_ranges + full_ranges // (2 * new_ranges))
        if sensor_data.ranges[new_i] == float('Inf') or np.isinf(sensor_data.ranges[new_i]):
            discretized_ranges.append(float('Inf'))
        elif np.isnan(sensor_data.ranges[new_i]):
            discretized_ranges.append(float('Nan'))
        else:
            discretized_ranges.append(sensor_data.ranges[new_i])
    return discretized_ranges

def init_quad(x, vx, z, vz, phi, w):
    # init function is for value iteration
    # quad model for value iteration is also 6D, so we need init all six state variables
    quad_pose = Pose()
    quad_pose.position.x = x
    quad_pose.position.y = 0
    quad_pose.position.z = z

    quad_twist = Twist()
    quad_twist.linear.x = vx
    quad_twist.linear.y = 0
    quad_twist.linear.z = vz

    quad_twist.angular.x = 0
    quad_twist.angular.y = w
    quad_twist.angular.z = 0

    qu_x, qu_y, qu_z, qu_w = quaternion_from_euler(0, phi, 0)
    quad_pose.orientation.x = qu_x
    quad_pose.orientation.y = qu_y
    quad_pose.orientation.z = qu_z
    quad_pose.orientation.w = qu_w

    quad_state = ModelState()
    quad_state.model_name = "quadrotor"
    quad_state.pose = quad_pose
    quad_state.twist = quad_twist

    rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)(quad_state)
    return True





if __name__ == "__main__":
    from quick_trainer import *
    # vf_model = load_model(os.environ['PROJ_HOME_3'] + '/tf_model/dubinsCar/vf_mpc.h5', custom_objects={'customerized_loss': customerized_loss})

    # vf_model = load_model(os.environ['PROJ_HOME_3'] + '/tf_model/dubinsCar/vf_mpc_new.h5')
    # csv_reader_path = os.environ['PROJ_HOME_3'] + '/data/dubinsCar/env_difficult/valFunc_mpc_filled_cleaned.csv'
    # csv_writer_path = os.environ['PROJ_HOME_3'] + '/data/dubinsCar/env_difficult/test_valFunc_mpc_filled_cleaned.csv'

    # vf_model = load_model(os.environ['PROJ_HOME_3'] + '/tf_model/dubinsCar/vf.h5')
    # csv_reader_path = os.environ['PROJ_HOME_3'] + '/data/dubinsCar/env_difficult/valFunc_filled_cleaned.csv'
    # csv_writer_path = os.environ['PROJ_HOME_3'] + '/data/dubinsCar/env_difficult/test_valFunc_filled_cleaned.csv'


    vf_model = load_model(os.environ['PROJ_HOME_3'] + '/tf_model/quad/vf_mpc.h5')
    csv_reader_path = os.environ['PROJ_HOME_3'] + '/data/quad/valFunc_mpc_filled_cleaned.csv'
    csv_writer_path = os.environ['PROJ_HOME_3'] + '/tests/test_value_prediction/temp.csv'

    # --- prepare stats for proper normalization --- #
    assert os.path.exists(csv_reader_path)

    colnames = ['x', 'vz', 'z', 'vz', 'phi', 'w', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'reward', 'value', 'cost', 'collision_in_future', 'collision_current', 'col_trajectory_flag']
    # colnames = ['x','y','theta','value','d1','d2','d3','d4','d5','d6','d7','d8']
    val_filled = pd.read_csv(csv_reader_path, names=colnames, na_values="?", comment='\t', sep=",", skipinitialspace=True, skiprows=1)

    stats_source = val_filled.copy()
    stats_source.dropna()
    stats = stats_source.describe()

    # if you use quadrotor data
    stats.pop("reward")
    stats.pop("value")
    stats.pop("cost")
    stats.pop("collision_in_future")
    stats.pop("collision_current")
    stats.pop("col_trajectory_flag")
    stats = stats.transpose()

    # if you use dubinscar data
    # stats.pop('value')
    # stats = stats.transpose()

    # input of vf model if you use quadrotor data
    val_filled.pop('reward')
    val_gt = val_filled.pop('value')
    val_filled.pop('cost')
    val_filled.pop('collision_in_future')
    val_filled.pop('collision_current')
    val_filled.pop('col_trajectory_flag')
    norm_val_filled = norm(val_filled, stats)

    # input of vf model if you use dubinscar data
    # val_gt = val_filled.pop('value')
    # norm_val_filled = norm(val_filled, stats)


    # writing dataset
    write_dataset = val_filled.copy()
    val_pred = np.empty(val_filled.shape[0], dtype=float)

    idx = 0
    for _, row in norm_val_filled.iterrows():
        tmp = vf_model.predict(np.array(row).reshape(1, -1))[0][0]
        val_pred[idx] = tmp
        idx += 1

    write_dataset['val_pred'] = val_pred
    write_dataset['val_gt']   = val_gt
    print(write_dataset.head())
    write_dataset.to_csv(csv_writer_path)

    print("everything is done!")
