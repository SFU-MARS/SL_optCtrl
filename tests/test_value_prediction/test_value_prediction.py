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

# if __name__ == "__main__":
#     vf_model = load_model(os.environ['PROJ_HOME_3'] + '/tf_model/quad/vf_merged.h5')
#
#     csv_reader_path = os.environ['PROJ_HOME_3'] + '/data/quad/valFunc_mpc_filled.csv'
#     csv_writer_path = os.environ['PROJ_HOME_3'] + '/tests/test_value_prediction/test_valFunc_prediction.csv'
#
#     # --- prepare stats for proper normalization --- #
#     val_filled_path = os.environ['PROJ_HOME_3'] + "/data/quad/valFunc_filled.csv"
#     val_filled_mpc_path = os.environ['PROJ_HOME_3'] + "/data/quad/valFunc_mpc_filled.csv"
#     assert os.path.exists(val_filled_mpc_path) and os.path.exists(val_filled_path)
#     colnames = ['x', 'vx', 'z', 'vz', 'phi', 'w', 'value', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
#     val_filled = pd.read_csv(val_filled_path, names=colnames, na_values="?", comment='\t', sep=",",
#                              skipinitialspace=True, skiprows=1)
#     val_filled_mpc = pd.read_csv(val_filled_mpc_path, names=colnames, na_values="?", comment='\t', sep=",",
#                                  skipinitialspace=True, skiprows=1)
#
#     stats_source = pd.concat([val_filled.copy(), val_filled_mpc.copy()])
#     stats_source.dropna()
#     stats = stats_source.describe()
#     stats = stats.transpose()
#
#     # --- prepare original MPC dataset --- #
#     raw_dataset = pd.read_csv(csv_reader_path, names=colnames, na_values="?", comment='\t', sep=",",
#                                  skipinitialspace=True, skiprows=1)
#     dataset = raw_dataset.copy()
#     unnormed_value_gt = dataset['value']
#
#     # --- normalize original MPC dataset --- #
#     norm_dataset = norm(dataset, stats)
#     print(norm_dataset.head())
#
#     # --- prepare write on new file row by row --- #
#     write_dataset = norm_dataset.copy()
#     val = np.empty(write_dataset.shape[0], dtype = float)
#     print(val.shape)
#
#     norm_dataset.pop('value')
#     write_dataset.pop('value')
#     print(norm_dataset.head())
#     i = 0
#
#     for idx, row in norm_dataset.iterrows():
#         # print("row:", np.array(row).reshape(1,-1))
#         # val = vf_model.predict(np.array(row).reshape(1,-1))
#         temp = vf_model.predict(np.array(row).reshape(1, -1))[0][0]
#         # print(temp, idx)
#         val[i] = temp
#         # write_dataset[idx]['value'] = val
#         # new_row = pd.DataFrame(np.array(list(row).append(val)))
#         i += 1
#
#     write_dataset['val_predict'] = val
#     write_dataset['val_gt'] = unnormed_value_gt
#     print(write_dataset.head())
#     write_dataset.to_csv(csv_writer_path)

# if __name__ == "__main__":
#     rospy.init_node("value prediction testing", anonymous=True, log_level=rospy.INFO)
#     # --- load value model --- #
#     vf_model = load_model(os.environ['PROJ_HOME_3'] + '/tf_model/quad/vf_merged.h5')
#     # --- prepare stats for proper normalization --- #
#     val_filled_path = os.environ['PROJ_HOME_3'] + "/data/quad/valFunc_filled.csv"
#     val_filled_mpc_path = os.environ['PROJ_HOME_3'] + "/data/quad/valFunc_mpc_filled.csv"
#     assert os.path.exists(val_filled_mpc_path) and os.path.exists(val_filled_path)
#     colnames = ['x', 'vx', 'z', 'vz', 'phi', 'w', 'value', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
#     val_filled = pd.read_csv(val_filled_path, names=colnames, na_values="?", comment='\t', sep=",",
#                              skipinitialspace=True, skiprows=1)
#     val_filled_mpc = pd.read_csv(val_filled_mpc_path, names=colnames, na_values="?", comment='\t', sep=",",
#                                  skipinitialspace=True, skiprows=1)
#
#     stats_source = pd.concat([val_filled.copy(), val_filled_mpc.copy()])
#     stats_source.dropna()
#     stats_source.pop('value')
#     stats = stats_source.describe()
#     stats = stats.transpose()
#
#     print("stats mean:", np.array(stats['mean']))
#     print("stats std:", np.array(stats['std']))
#
#     # --- sample single point in gazebo --- #
#
#     # --- (1) first reset gazebo --- #
#     rospy.wait_for_service('/gazebo/reset_simulation')
#     print("# do I get here??")
#     try:
#         rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
#     except rospy.ServiceException as e:
#         print("# Reset simulation failed!")
#
#     x, vx, z, vz, phi, w = -3, 2, 5.0, 2, 0.19, 0
#     # print("current state:", [x,vx,z,vz,phi,w])
#     init_quad(x, vx, z, vz, phi, w)
#
#     # --- (2) obtain sensor readings --- #
#     sensor_data = None
#     # take an instant unfreezing
#     rospy.wait_for_service('/gazebo/unpause_physics')
#     try:
#         rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
#     except rospy.ServiceException as e:
#         print("/gazebo/unpause_physics service call failed")
#     # Receive sensor data
#     while sensor_data is None:
#         rospy.wait_for_service("/gazebo/get_model_state")
#         try:
#             sensor_data = rospy.wait_for_message('/scan', LaserScan, timeout=10)
#         except rospy.ServiceException as e:
#             print("/gazebo/get_model_state service call failed!")
#     # pause simulation to prepare sample
#     rospy.wait_for_service('/gazebo/pause_physics')
#     try:
#         rospy.ServiceProxy('/gazebo/pause_physics', Empty)
#     except rospy.ServiceException as e:
#         print("/gazebo/pause_physics service call failed")
#
#     #  ignore any invalid sensor readings
#     discrete_sensor_data = discretize_sensor(sensor_data, 8)
#     if np.isnan(discrete_sensor_data).any() or np.isinf(discrete_sensor_data).any():
#         raise ValueError("invalid sensor reading !!")
#
#
#     # --- test value of single data using trained value model --- #
#     sample = [x, vz, z, vz, phi, w] + discrete_sensor_data
#     normed_sample = norm(sample, stats)
#     print("sample:", sample)
#     print("normed_sample:", normed_sample)
#     val_res = vf_model.predict(np.array(normed_sample).reshape(1, -1))[0][0]


    print("cur value result:", val_res)

if __name__ == "__main__":
    vf_model = load_model(os.environ['PROJ_HOME_3'] + '/tf_model/dubinsCar/vf.h5')

    csv_reader_path = os.environ['PROJ_HOME_3'] + '/data/dubinsCar/valFunc_filled_cleaned.csv'
    csv_writer_path = os.environ['PROJ_HOME_3'] + '/data/dubinsCar/test_valFunc_filled_cleaned.csv'

    # --- prepare stats for proper normalization --- #
    val_filled_path = os.environ['PROJ_HOME_3'] + "/data/dubinsCar/valFunc_filled_cleaned.csv"
    assert os.path.exists(val_filled_path)
    colnames = ['x', 'y', 'theta', 'value', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8']
    val_filled = pd.read_csv(val_filled_path, names=colnames, na_values="?", comment='\t', sep=",",
                             skipinitialspace=True, skiprows=1)

    stats_source = val_filled.copy()
    stats_source.dropna()
    stats = stats_source.describe()
    stats = stats.transpose()

    # input of vf model
    norm_val_filled = norm(val_filled, stats)
    norm_val_filled.pop('value')


    # writing dataset
    write_dataset = val_filled.copy()
    val_series = np.empty(val_filled.shape[0], dtype=float)


    for idx, row in norm_val_filled.iterrows():
        temp = vf_model.predict(np.array(row).reshape(1, -1))[0][0]
        val_series[idx] = temp
        idx += 1

    write_dataset['val_predict'] = val_series
    print(write_dataset.head())
    write_dataset.to_csv(csv_writer_path)

