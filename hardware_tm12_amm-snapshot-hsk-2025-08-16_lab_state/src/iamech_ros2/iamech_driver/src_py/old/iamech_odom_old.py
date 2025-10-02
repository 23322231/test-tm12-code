#!/usr/bin/env python3
# Version 3.0
#Python package
import signal
import pyads
import threading
import math

# ROS 
import rospy
import tf

# ROS msg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3

# user-define msg
from iamech_ros.srv import WPLC, WPLCResponse
from iamech_ros.msg import PLCStatus

class iAmechROS:
    def __init__(self):
        # threading.Thread.__init__(self)
        ## Basic parameters
        self.position = [0., 0., 0., 0., 0., 0.] # AMM current position.
        self.L_mm = rospy.get_param("~wheelbase", 588.0) # Wheelbase
        self.send_tf = rospy.get_param("~send_tf", True) # send tf (/odom->/baselink) or not
        
        ## ROS Publisher
        self.tf_broadcaster = tf.TransformBroadcaster() # send tf (/odom->/baselink)
        self.odom_pub = rospy.Publisher("/odom", Odometry, queue_size=5) # send Qdometry msg
        
        ## Initalize odom msg
        self.odom = Odometry()
        self.odom.header.frame_id = "/odom"
        self.odom.child_frame_id = "/base_link"

        self.last_status = None

        ## Odom type
        odom_type = rospy.get_param("~odom_type", "shang")
        if odom_type == "jessica":
            rospy.Subscriber("/Manager/plc_status", PLCStatus, self._odom_jessica, queue_size=1)
        else:
            rospy.Subscriber("/Manager/plc_status", PLCStatus, self._callback, queue_size=1)

    def _odom_pub(self, now, twist):
        quat = Quaternion()
        quat.x = 0.0
        quat.y = 0.0
        quat.z = math.sin(self.position[5] / 2.0)
        quat.w = math.cos(self.position[5] / 2.0)
        
        if self.send_tf:
            self.tf_broadcaster.sendTransform((self.position[0], self.position[1], self.position[2]), (quat.x, quat.y, quat.z, quat.w), now, "/base_link", "/odom")

        ## Publish Odom 
        self.odom.header.stamp = now
        self.odom.pose.pose.position.x = self.position[0]
        self.odom.pose.pose.position.y = self.position[1]
        self.odom.pose.pose.orientation = quat

        self.odom.twist.twist = twist

        self.odom_pub.publish(self.odom)
    
    def _odom_jessica(self, req):
        if self.last_status is None:
            self.last_status = req.header.stamp
            return
        
        now = req.header.stamp

        delta_t = now - self.last_status
        delta_t = delta_t.to_sec()

        vel_x = (req.left.velocity + req.left.velocity ) * math.cos(self.position[5]) / 2000.0
        vel_y = (req.left.velocity  + req.left.velocity ) * math.sin(self.position[5]) / 2000.0
        vel_w = 1.0 / self.L_mm * (req.right.velocity  - req.left.velocity)

        self.position[0] += vel_x * delta_t
        self.position[1] += vel_y * delta_t
        self.position[5] += vel_w * delta_t
        self.position[5] %= (math.pi * 2)

        self._odom_pub(now, Twist(Vector3(vel_x, vel_y, 0), Vector3(0, 0, vel_w)))
        
        self.last_status = now
    
    def _callback(self, req):
        # diff_left_mm = curr[0] - last[0]
        # diff_right_mm = curr[1] - last[1]
        if self.last_status is None:
            self.last_status = req
            return
        
        now = req.header.stamp

        diff_left_mm = req.left.pos - self.last_status.left.pos
        diff_right_mm = req.right.pos - self.last_status.right.pos

        self._compute_pose(diff_right_mm, diff_left_mm)

        vel_x = (req.left.velocity + req.left.velocity ) * math.cos(self.position[5]) / 2000.0
        vel_y = (req.left.velocity  + req.left.velocity ) * math.sin(self.position[5]) / 2000.0
        vel_w = 1.0 / self.L_mm * (req.right.velocity  - req.left.velocity)

        self._odom_pub(now, Twist(Vector3(vel_x, vel_y, 0), Vector3(0, 0, vel_w)))
        
        self.last_status = req

    def _compute_pose(self, diff_right_mm, diff_left_mm):
        ### Doesn't move
        if diff_left_mm == 0 and diff_right_mm == 0: return
        dx_mm = 0;dy_mm = 0;dw_rad = 0
        ### Straight
        if diff_right_mm == diff_left_mm:
            self.position[0] += diff_right_mm * math.cos(self.position[5]) / 1000
            self.position[1] += diff_right_mm * math.sin(self.position[5]) / 1000
            return
        ### Spin in place
        elif diff_right_mm + diff_left_mm == 0:
            self.position[5] = (self.position[5] + 2.0 * diff_right_mm / self.L_mm) % (math.pi * 2)
            return
        else:
            ### Turn Left
            if abs(diff_left_mm) < abs(diff_right_mm):
                theta = (diff_right_mm - diff_left_mm) / float(self.L_mm)
                temp = self.L_mm / 2.0
                if diff_left_mm != 0:
                    temp += diff_left_mm / theta
                dw_rad = theta
                dx_ = temp * math.sin(theta)
                dy_ = temp * (1 - math.cos(theta))
            ### Trun Right
            else:
                theta = (diff_left_mm - diff_right_mm) / float(self.L_mm)
                temp = self.L_mm / 2.0
                if diff_right_mm != 0:
                    temp += diff_right_mm / theta
                dw_rad = -theta
                dx_ = temp * math.sin(theta)
                dy_ = temp * (math.cos(theta) - 1)
            
            dx_mm = dx_ * math.cos(self.position[5]) - dy_ * math.sin(self.position[5])
            dy_mm = dx_ * math.sin(self.position[5]) + dy_ * math.cos(self.position[5])

            self.position[0] += dx_mm / 1000
            self.position[1] += dy_mm / 1000
            self.position[5] =  (self.position[5] + dw_rad) % (math.pi * 2) 

def SIGINT_handler(signum, frame):
    rospy.loginfo("Keyboard Interrup")
    rospy.signal_shutdown('keyboard interrupt')

if __name__ == '__main__':
    rospy.init_node('iAmech_odometry', log_level=rospy.DEBUG)
    signal.signal(signal.SIGINT, SIGINT_handler)
    iAmechROS()
    rospy.spin()
    rospy.loginfo("End of driver node")