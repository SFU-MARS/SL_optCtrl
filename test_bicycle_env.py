from geometry_msgs.msg import Twist, Pose
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import GetLinkState
from gazebo_msgs.srv import SetLinkState
from gazebo_msgs.srv import SpawnModel
from ackermann_msgs.msg import AckermannDriveStamped


import rospy
from tf.transformations import euler_from_quaternion
import time
import numpy as np
import sys


if __name__ == "__main__":
    # You have to initialize node at first when using rospy.
    # the node name could be set as you wish.
    # Actually the node here means your own code file
    rospy.init_node("test_ackermann", anonymous=True, log_level=rospy.INFO)
    srv_unpause = rospy.ServiceProxy('/ackermann_vehicle/gazebo/unpause_physics', Empty)
    srv_pause = rospy.ServiceProxy('/ackermann_vehicle/gazebo/pause_physics', Empty)
    srv_reset_proxy = rospy.ServiceProxy('/ackermann_vehicle/gazebo/reset_simulation', Empty)
    srv_get_model_state = rospy.ServiceProxy('/ackermann_vehicle/gazebo/get_model_state', GetModelState)
    srv_set_model_state = rospy.ServiceProxy('/ackermann_vehicle/gazebo/set_model_state', SetModelState)
    srv_get_link_state  = rospy.ServiceProxy('/ackermann_vehicle/gazebo/get_link_state', GetLinkState)
    srv_set_link_state  = rospy.ServiceProxy('/ackermann_vehicle/gazebo/set_link_state', SetLinkState)

    ack_publisher = rospy.Publisher('/ackermann_vehicle/ackermann_cmd', AckermannDriveStamped, queue_size=5)

    # ----- reset simulator -------
    rospy.wait_for_service('/ackermann_vehicle/gazebo/reset_simulation')
    print("do I get here??")
    try:
        srv_reset_proxy()
    except rospy.ServiceException as e:
        print("# Reset simulation failed!")

    while True:

        # ------ unpause simulator to run ------
        rospy.wait_for_service('/ackermann_vehicle/gazebo/unpause_physics')
        try:
            srv_unpause()
        except rospy.ServiceException as e:
            print("# /gazebo/unpause_physics service call failed")

        ack_msg = AckermannDriveStamped()
        # ack_msg.header.stamp = rospy.Time.now()
        # ack_msg.header.frame_id = ''
        ack_msg.drive.steering_angle = -0.75
        ack_msg.drive.steering_angle_velocity = 0
        ack_msg.drive.speed = 1.0
        ack_publisher.publish(ack_msg)

        # ------ retrieve link state we want ------
        left_link_state = None
        right_link_state = None
        rospy.wait_for_service("/ackermann_vehicle/gazebo/get_link_state")
        try:
            left_link_state = srv_get_link_state(link_name='left_front_wheel', reference_frame='base_link')
            right_link_state = srv_get_link_state(link_name='right_front_wheel', reference_frame='base_link')
        except rospy.ServiceException as e:
            print("# /gazebo/get_link_state service call failed")



        rospy.wait_for_service('/ackermann_vehicle/gazebo/pause_physics')
        try:
            srv_pause()
        except rospy.ServiceException as e:
            print("# /gazebo/pause_physics service call failed")

        lox = left_link_state.link_state.pose.orientation.x
        loy = left_link_state.link_state.pose.orientation.y
        loz = left_link_state.link_state.pose.orientation.z
        low = left_link_state.link_state.pose.orientation.w
        # axis: sxyz
        _, _, lyaw = euler_from_quaternion([lox, loy, loz, low])

        rox = right_link_state.link_state.pose.orientation.x
        roy = right_link_state.link_state.pose.orientation.y
        roz = right_link_state.link_state.pose.orientation.z
        row = right_link_state.link_state.pose.orientation.w

        _, _, ryaw = euler_from_quaternion([rox, roy, roz, row])

        if lyaw > 1:
            lyaw = lyaw - np.pi
        elif lyaw < -1:
            lyaw = lyaw + np.pi

        if ryaw > 1:
            ryaw = ryaw - np.pi
        elif ryaw < -1:
            ryaw = ryaw + np.pi



        print("steering angle:", (lyaw + ryaw)/2)
        # print("right yaw:", ryaw)

