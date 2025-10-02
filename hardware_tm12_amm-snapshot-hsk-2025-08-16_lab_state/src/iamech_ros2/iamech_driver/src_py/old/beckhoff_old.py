#!/usr/bin/env python3
# Version 3.0 by shang
#Python package
import pyads

# ROS 
import rospy

# ROS msg
from geometry_msgs.msg import Twist

# user-define msg
from iamech_ros.srv import WPLC, WPLCResponse
from iamech_ros.msg import PLCStatus

class beckhoff_connect:
    def __init__(self) -> None:
        PLC_AMS_ID = rospy.get_param("~plc_ams_id",'192.168.100.100.1.1')
        PLC_IP = rospy.get_param("~plc_ip",'192.168.100.100')
        SENDER_AMS = rospy.get_param("~sender_ams",'1.2.3.4.1.1')
        pyads.set_local_address(SENDER_AMS)
        self.plc = pyads.Connection(PLC_AMS_ID, 801, PLC_IP)

        while not rospy.is_shutdown():
            try:
                self.plc.open()
                rospy.logdebug("[Connect] PLC")
                break
            except:
                rospy.logerr("Waiting for PLC connection")
    
    def read_by_name(self, name):
        try:
            result = self.plc.read_by_name(name)
        except:
            rospy.logerr("[Disconnect] PLC")
            result = None
        return result

    def read_list_by_name(self, name):
        try:
            result = self.plc.read_list_by_name(name)
        except:
            rospy.logerr("[Disconnect] PLC")
            result = None
        return result
    
    def write_by_name(self, name, value):
        try:
            self.plc.write_by_name(name, value)
        except:
            rospy.logerr("[Disconnect] PLC")

    def write_list_by_name(self, name_and_value:dict):
        try:
            self.plc.write_list_by_name(name_and_value)
        except:
            rospy.logerr("[Disconnect] PLC")

class service_handler:
    def __init__(self) -> None:
        self.plc = beckhoff_connect()

        rospy.Service('~write', WPLC, self.write_service_callback)
    
    def write_service_callback(self, req):
        self.plc.write_by_name(req.name, req.value)
        return WPLCResponse()

class cmd_vel_handler:
    def __init__(self) -> None:

        self.L_mm = rospy.get_param("~wheelbase", 588.0)
        self.v_mm_s = {
            ".SLAM_L[2]": 0,
            ".SLAM_R[2]": 0
        }

        self.plc = beckhoff_connect()
        
        rospy.Subscriber("/cmd_vel", Twist, self.cmd_vel_callback)
    
    def __del__(self) -> None:
        try:
            self.plc.close()
        except:
            pass

    def cmd_vel_callback(self, req):
        rospy.loginfo(f'Get new cmd_vel ({req.linear.x} , {req.angular.z})')

        if req.linear.x == 0: temp = 0
        else: temp = 2 * req.linear.x * 1000

        if req.angular.z == 0: temp2 = 0
        else: temp2 = req.angular.z * self.L_mm

        self.v_mm_s[".SLAM_R[2]"] = int((temp + temp2) / 2)
        self.v_mm_s[".SLAM_L[2]"] = int((temp - temp2) / 2)

        self.plc.write_list_by_name(self.v_mm_s)

class pub_plc_handler:
    def __init__(self) -> None:
        self.parameters_list =[
            ".bSLAM_ServeON",
            ".SLAM_R[11]",
            ".SLAM_R[12]",
            ".SLAM_R[13]",
            ".SLAM_R[14]",
            ".SLAM_R[15]",
            ".SLAM_R[16]",
            ".SLAM_R[17]",
            ".SLAM_R[18]",
            ".SLAM_L[11]",
            ".SLAM_L[12]",
            ".SLAM_L[13]",
            ".SLAM_L[14]",
            ".SLAM_L[15]",
            ".SLAM_L[16]",
            ".SLAM_L[17]",
            ".SLAM_L[18]"
        ]

        self.status = PLCStatus()

        self.plc = beckhoff_connect()

        if rospy.get_param("~auto_serveon", False):
            self.plc.write_by_name(".bSLAM_ServeON", 1)

        self.pub = rospy.Publisher("~plc_status", PLCStatus, queue_size=50)
    
    def __del__(self):
        if rospy.get_param("~auto_serveon", False):
            self.plc.write_by_name(".bSLAM_ServeON", 0)

    def poll(self):
        rate = rospy.Rate(rospy.get_param("~fps", 60))

        while not rospy.is_shutdown():
            param = self.plc.read_list_by_name(self.parameters_list)
            if param is None: continue
            
            self.status.ServeON = param[self.parameters_list[0]]
            self.status.right.bReady = param[self.parameters_list[1]]
            self.status.right.bMoving = param[self.parameters_list[2]]
            self.status.right.bError = param[self.parameters_list[3]]
            self.status.right.pos = param[self.parameters_list[4]]
            self.status.right.velocity = param[self.parameters_list[5]]
            self.status.right.ErrorCode = param[self.parameters_list[6]]
            self.status.right.temperature = param[self.parameters_list[7]]
            self.status.right.volt = param[self.parameters_list[8]]

            self.status.left.bReady = param[self.parameters_list[9]]
            self.status.left.bMoving = param[self.parameters_list[10]]
            self.status.left.bError = param[self.parameters_list[11]]
            self.status.left.pos = param[self.parameters_list[12]]
            self.status.left.velocity = param[self.parameters_list[13]]
            self.status.left.ErrorCode = param[self.parameters_list[14]]
            self.status.left.temperature = param[self.parameters_list[15]]
            self.status.left.volt = param[self.parameters_list[16]]

            self.status.header.stamp = rospy.Time.now()

            self.pub.publish(self.status)

            rate.sleep()


if __name__=="__main__":
    rospy.init_node('Manager', log_level=rospy.DEBUG) # ROS Node init.

    worker = []
    worker.append(service_handler())
    worker.append(cmd_vel_handler())

    pub_worker = pub_plc_handler()

    
    pub_worker.poll()