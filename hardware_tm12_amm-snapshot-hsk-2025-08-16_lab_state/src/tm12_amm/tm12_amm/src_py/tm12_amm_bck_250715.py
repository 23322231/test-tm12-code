#!/usr/bin/env python3

import sys
import os
import numpy as np                                     # Python数值计算库
import cv2                                             # Opencv图像处理库
import time
import threading
import math
from math import pi
from typing import TypedDict, Optional
import numpy.typing as npt
import signal
import logging
import asyncio
import uuid
import json

import rclpy
import rclpy.callback_groups
import rclpy.parameter
from rclpy.task import Future
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, \
								ReentrantCallbackGroup
from rclpy.action import ActionServer, \
				CancelResponse, \
				GoalResponse
from rclpy.action.server import ServerGoalHandle

import ros2_numpy as rnp

from cv_bridge import CvBridge                         # ROS与OpenCV图像转换类


from rcl_interfaces.msg import ParameterDescriptor, \
				SetParametersResult
from std_msgs.msg import Header, Char, String                   # ROS消息类型
from geometry_msgs.msg import Twist, Pose, Quaternion   # ROS消息类型
from sensor_msgs.msg import Image, PointCloud2

from realsense2_camera_msgs.msg import RGBD
from tm_msgs.msg import FeedbackState

from std_srvs.srv import Trigger
from tm_msgs.srv import SetPositions
from grpr2f85_ifaces.srv import SetGripperState, GetGripperStatus

from tm12_amm_interfaces.action import Dotask, Calibration

from ai_interfaces.srv import Infer, Success # 您的服務消息類型

# 導入校正模塊
from .calibration import TM12Calibration


class TM12_AMM_ROS2_Node(Node):
	def __init__(self, name='tm12_amm'):
		super().__init__(name)
		self.get_logger().info('TM12_AMM_ROS2_Node_py init')
		self.cv_bridge = CvBridge()
		# Initialize timer

		self.set_parameter()
		self.set_callback_group()
		self.set_subscriber()
		self.set_publisher()
		self.set_service_client()
		self.set_service_server()
		self.set_action_client()
		self.set_action_server()

		self.timer = self.create_timer(1.0, self.timer_callback, callback_group=self.Reentrant_cb_group)

		self.parameter_callback_handle = self.add_on_set_parameters_callback(self.parameter_callback)

		self.action_map = {
			"Calibration": self.calibration,
			"AI_Action": self.ai_action
		}

	def set_parameter(self):#ok
		read_only = ParameterDescriptor(read_only=True)

		# 宣告唯讀參數
		self.z_min_ = self.declare_parameter('z_min', 0.035, read_only).value
		self.velocity_max_ = self.declare_parameter('velocity_max', 2.0, read_only).value
		self.acc_time_min_ = self.declare_parameter('acc_time_min', 0.2, read_only).value
		self.home_in_joint_ = self.declare_parameter('home_in_joint', [-pi/4, 0.0, pi/2, 0.0, pi/2, 0.0], read_only).value

		# 相機參數宣告與轉換
		camera_matrix_list = self.declare_parameter('camera.matrix', [
			1.0, 0.0, 0.0,
			0.0, 1.0, 0.0,
			0.0, 0.0, 1.0
		], read_only).value
		self.camera_matrix_ = np.array(camera_matrix_list, dtype=np.float64).reshape(3, 3)
		#self.get_logger().info(f'Camera matrix:\n{str(self.camera_matrix_)}')

		dist_coeffs_list = self.declare_parameter('camera.distortion', [0.0, 0.0, 0.0, 0.0, 0.0], read_only).value
		self.dist_coeffs_ = np.array(dist_coeffs_list, dtype=np.float64)
		#self.get_logger().info(f'Distortion coefficients:\n{str(self.dist_coeffs_)}')


		'''
		# 計算新的相機矩陣用於校正
		image_size = (1280, 720)  # 根據實際相機解析度調整
		self.new_camera_matrix_, self.roi = cv2.getOptimalNewCameraMatrix(
			self.camera_matrix_,
			self.dist_coeffs_,
			image_size,
			1,
			image_size
		)

		# 預先計算重投影映射
		self.mapx_, self.mapy_ = cv2.initUndistortRectifyMap(
			self.camera_matrix_,
			self.dist_coeffs_,
			None,
			self.new_camera_matrix_,
			image_size,
			cv2.CV_32FC1
		)
		'''
		# 手眼校正參數宣告與轉換
		R_c2g_list = self.declare_parameter('hand_eye.rotation', [
		1.0, 0.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 0.0, 1.0
		], read_only).value
		self.R_c2g_ = np.array(R_c2g_list, dtype=np.float64).reshape(3, 3)

		t_c2g_list = self.declare_parameter('hand_eye.translation', [0.0, 0.0, 0.0], read_only).value
		self.t_c2g_ = np.array(t_c2g_list, dtype=np.float64)

		self.T_Cam2Grpr_ = np.eye(4)
		self.T_Cam2Grpr_[:3, :3] = self.R_c2g_
		self.T_Cam2Grpr_[:3, 3] = self.t_c2g_

		R_c2g_offset_list = self.declare_parameter('hand_eye_offset.rotation', [
		1.0, 0.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 0.0, 1.0
		], read_only).value
		self.R_c2g_offset_ = np.array(R_c2g_offset_list, dtype=np.float64).reshape(3, 3)

		t_c2g_offset_list = self.declare_parameter('hand_eye_offset.translation', [0.0, 0.0, 0.0], read_only).value
		self.t_c2g_offset_ = np.array(t_c2g_offset_list, dtype=np.float64)

		self.T_Cam2Grpr_offset_ = np.eye(4)
		self.T_Cam2Grpr_offset_[:3, :3] = self.R_c2g_offset_
		self.T_Cam2Grpr_offset_[:3, 3] = self.t_c2g_offset_

		#self.get_logger().info(f'Hand-eye Tansform:\n{str(self.T_Cam2Grpr_offset_)}')

		self.manual_mode_ = self.declare_parameter('manual_mode', False).value
		self.pose_take_photo_static_ = self.declare_parameter('pose_take_photo_static', [0.36, -0.44, 0.61, -pi, 0.0, pi/4]).value

		# 手動操作速度控制參數
		self.manual_speed_ = self.declare_parameter('manual_speed', 5.0).value
		self.manual_speed_min_ = self.declare_parameter('manual_speed_min', 1.0).value
		self.manual_speed_max_ = self.declare_parameter('manual_speed_max', 20.0).value
		self.manual_speed_step_ = self.declare_parameter('manual_speed_step', 1.0).value

		self.rgb_img_ = None
		self.depth_img_ = None
		self.tm12_feedback_ = None

	def set_callback_group(self):# ok
		self.MutuallyExclusive_cb_group = MutuallyExclusiveCallbackGroup()
		self.Reentrant_cb_group = ReentrantCallbackGroup()

	def set_subscriber(self):
		self.realsense_rgb_subscription = self.create_subscription(
			Image,
			"arm/camera/realsense_camera/color/image_raw",
			self.realsense_rgb_callback,
			10,
			callback_group=self.Reentrant_cb_group
		)
		self.realsense_depth_subscription = self.create_subscription(
			Image,
			"arm/camera/realsense_camera/aligned_depth_to_color/image_raw",
			self.realsense_depth_callback,
			10,
			callback_group=self.Reentrant_cb_group
		)
		self.realsense_ptcloud_subscription = self.create_subscription(
		 	PointCloud2,
		 	"arm/camera/realsense_camera/depth/color/points",
			self.realsense_ptcloud_callback,
		 	10,
		 	callback_group=self.Reentrant_cb_group
		)
		# Initialize feedback subscription and service
		self.tm12_feedback_subscription = self.create_subscription(
			FeedbackState,
			"arm/tm12/tm_driver/feedback_states",
			self.tm12_feedback_callback,
			10,
			callback_group=self.Reentrant_cb_group
		)
		# Initialize keyboard manual subscription and parameter callback
		self.keyboard_manual_subscription = self.create_subscription(
			Char,
			'keyboard_manual',
			self.keyboard_manual_callback,
			10)

	def set_publisher(self):#ok
		# AMR 移動控制發布器
		self.amr_twist_publisher = self.create_publisher(
			Twist,
			'amr/iamech/iamech_driver/cmd_vel',
			10
		)

	def set_service_client(self):#ok
		# gripper 2f85 clients
		self.grpr2f85_set_gripper_state_client = self.create_client(
			SetGripperState,
			'arm/gripper/grpr2f85_driver/set_gripper_state',
        		callback_group=self.MutuallyExclusive_cb_group
		)
		self.grpr2f85_get_gripper_status_client = self.create_client(
			GetGripperStatus,
			'arm/gripper/grpr2f85_driver/get_gripper_status',
        		callback_group=self.MutuallyExclusive_cb_group
		)
		# iaMech clients
		self.amr_servo_on_client = self.create_client(
			Trigger,
			'amr/iamech/iamech_driver/servo_on',
        		callback_group=self.MutuallyExclusive_cb_group
		)
		self.amr_servo_off_client = self.create_client(
			Trigger,
			'amr/iamech/iamech_driver/servo_off',
        		callback_group=self.MutuallyExclusive_cb_group
		)
		# TM12 clients
		self.tm12_set_positions_client = self.create_client(
			SetPositions,
			'arm/tm12/tm_driver/set_positions',
        		callback_group=self.MutuallyExclusive_cb_group
		)
		# AI inference client
		self.ai_inference_client = self.create_client(
			Infer,
			'ai_inference',
			callback_group=self.MutuallyExclusive_cb_group
		)
		self.report_success2ai_client = self.create_client(
			Success,
			'report_success',
			callback_group=self.MutuallyExclusive_cb_group
		)

	def set_service_server(self):
		# Initialize homing service
		self.homing = self.create_service(
			Trigger,
			'arm/homing',
			self.homing_callback,
			callback_group=self.Reentrant_cb_group  # 加入回調群組
		)
		self.take_picture = self.create_service(
			Trigger,
			'arm/take_picture',
			self.take_picture_callback,
			callback_group=self.Reentrant_cb_group  # 加入回調群組
		)
		# Initialize manual mode service
		self.enable_manual_mode = self.create_service(
			Trigger,
			'arm/enable_manual_mode',
			self.enable_manual_mode_callback,
			callback_group=self.Reentrant_cb_group  # 加入回調群組
		)
		self.disable_manual_mode = self.create_service(
			Trigger,
			'arm/disable_manual_mode',
			self.disable_manual_mode_callback,
			callback_group=self.Reentrant_cb_group  # 加入回調群組
		)

		pass

	def set_action_client(self):#ok
		pass

	def set_action_server(self):
		self.ai_action_server = ActionServer(
			self,
			Dotask,
			'arm/ai_action',
            self.execute_callback,
            callback_group=self.Reentrant_cb_group,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

	def parameter_callback(self, parameters):#ok
		"""
		處理參數更新的回調函數
		"""

		result = SetParametersResult(successful=True)

		try:
			for parameter in parameters:
				if parameter.name == 'manual_mode':
					# 更新手動模式狀態
					self.manual_mode_ = parameter.value
					self.get_logger().info(f'手動模式已{"啟用" if self.manual_mode_ else "停用"}')

					# 當啟用手動模式時，顯示幫助信息
					if self.manual_mode_:
						self.print_manual_mode_help()
					else:
						# 當停用手動模式時，停止 AMR 移動
						self.stop_amr()

				elif parameter.name == 'manual_speed':
					# 更新手動操作速度
					old_speed = self.manual_speed_
					self.manual_speed_ = max(self.manual_speed_min_,
						min(parameter.value, self.manual_speed_max_))
					self.get_logger().info(f'手動操作速度已從 {old_speed:.1f} 更新為 {self.manual_speed_:.1f}')

				elif parameter.name == 'pose_take_photo_static':
					new_pose = parameter.value
					if len(new_pose) != 6:
						raise ValueError("拍照位置必須包含 6 個元素 [x, y, z, rx, ry, rz]")
					self.pose_take_photo_static_ = new_pose
					self.get_logger().info(f'拍照位置已更新: {new_pose}')

		except Exception as e:
			self.get_logger().error(f'參數更新失敗: {str(e)}')
			result.successful = False
			result.reason = str(e)

		return result

####################################################

	def realsense_rgb_callback(self, msg: Image): #ok
		try:
			self.rgb_img_ = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
		except Exception as e:
			self.get_logger().error(f'Error in RGB image callback: {e}')

	def realsense_depth_callback(self, msg: Image): #ok
		try:
			self.depth_img_ = self.cv_bridge.imgmsg_to_cv2(msg, "32FC1")
			self.depth_img_ /= 1000.0
		except Exception as e:
			self.get_logger().error(f'Error in depth image callback: {e}')

	def realsense_ptcloud_callback(self, msg: PointCloud2): #ok
		try:
			self.ptcloud_data_ = rnp.numpify(msg)
		except Exception as e:
			self.get_logger().error(f'Error in PointCloud2 callback: {e}')

	def tm12_feedback_callback(self, msg: FeedbackState): #ok
		try:
			self.tm12_feedback_ = msg
		except Exception as e:
			self.get_logger().error(f'Error in TM12 feedback callback: {e}')

	def keyboard_manual_callback(self, msg: Char):
		"""處理鍵盤手動操作的回調函數"""
		if not self.manual_mode_:
			self.get_logger().warn('手動模式未啟用。請先調用 arm/enable_manual_mode 服務或按 "m" 鍵啟用手動模式。')
			return

		if self.tm12_feedback_ is None:
			self.get_logger().warn('TM12 feedback not available')
			return

		try:
			# 獲取按鍵字符
			key_char = chr(msg.data) if msg.data < 128 else str(msg.data)
			self.get_logger().debug(f'Received key: {key_char} (ASCII: {msg.data})')

			# 處理按鍵
			self.process_key(key_char)

		except Exception as e:
			self.get_logger().error(f'Error in keyboard manual callback: {e}')

	def process_key(self, key):
		"""處理鍵盤輸入的手動操作"""
		if self.tm12_feedback_ is None:
			self.get_logger().warn('TM12 feedback not available')
			return

		try:
			# 獲取當前機器人位置
			current_pose = self.tm12_feedback_.tool_pose.copy()
			speed = self.manual_speed_  # 使用可調整的速度

			# 根據按鍵執行相應操作
			if key == 'q':
				self.get_logger().info(f'Move +X (diagonal), speed: {speed}')
				current_pose[0] += speed * 0.001 * np.cos(-pi/4)
				current_pose[1] += speed * 0.001 * np.sin(-pi/4)
				self.call_tm12_set_positions(positions=current_pose)

			elif key == 'a':
				self.get_logger().info(f'Move -X (diagonal), speed: {speed}')
				current_pose[0] -= speed * 0.001 * np.cos(-pi/4)
				current_pose[1] -= speed * 0.001 * np.sin(-pi/4)
				self.call_tm12_set_positions(positions=current_pose)

			elif key == 'w':
				self.get_logger().info(f'Move +Y (diagonal), speed: {speed}')
				current_pose[0] += speed * 0.001 * (-np.sin(-pi/4))
				current_pose[1] += speed * 0.001 * np.cos(-pi/4)
				self.call_tm12_set_positions(positions=current_pose)

			elif key == 's':
				self.get_logger().info(f'Move -Y (diagonal), speed: {speed}')
				current_pose[0] -= speed * 0.001 * (-np.sin(-pi/4))
				current_pose[1] -= speed * 0.001 * np.cos(-pi/4)
				self.call_tm12_set_positions(positions=current_pose)

			elif key == 'e':
				self.get_logger().info(f'Move +Z (up), speed: {speed}')
				current_pose[2] += speed * 0.001
				self.call_tm12_set_positions(positions=current_pose)

			elif key == 'd':
				self.get_logger().info(f'Move -Z (down), speed: {speed}')
				current_pose[2] -= speed * 0.001
				self.call_tm12_set_positions(positions=current_pose)

			elif key == 'r':
				self.get_logger().info(f'Rotate +Roll, speed: {speed}')
				current_pose[3] += speed * pi / 1800
				self.call_tm12_set_positions(positions=current_pose)

			elif key == 'f':
				self.get_logger().info(f'Rotate -Roll, speed: {speed}')
				current_pose[3] -= speed * pi / 1800
				self.call_tm12_set_positions(positions=current_pose)

			elif key == 't':
				self.get_logger().info(f'Rotate +Pitch, speed: {speed}')
				current_pose[4] += speed * pi / 1800
				self.call_tm12_set_positions(positions=current_pose)

			elif key == 'g':
				self.get_logger().info(f'Rotate -Pitch, speed: {speed}')
				current_pose[4] -= speed * pi / 1800
				self.call_tm12_set_positions(positions=current_pose)

			elif key == 'y':
				self.get_logger().info(f'Rotate +Yaw, speed: {speed}')
				current_pose[5] += speed * pi / 1800
				self.call_tm12_set_positions(positions=current_pose)

			elif key == 'h':
				self.get_logger().info(f'Rotate -Yaw, speed: {speed}')
				current_pose[5] -= speed * pi / 1800
				self.call_tm12_set_positions(positions=current_pose)

			elif key == 'z':
				self.get_logger().info('Gripper close')
				self.call_grpr2f85_set_gripper_state(position=255)

			elif key == 'x':
				self.get_logger().info('Gripper open')
				self.call_grpr2f85_set_gripper_state(position=0)

			elif key == 'c':
				self.get_logger().info('AMR move forward')
				twist_msg = Twist()
				twist_msg.linear.x = 0.01
				twist_msg.angular.z = 0.0
				self.amr_twist_publisher.publish(twist_msg)

			elif key == 'v':
				self.get_logger().info('AMR move backward')
				twist_msg = Twist()
				twist_msg.linear.x = -0.01
				twist_msg.angular.z = 0.0
				self.amr_twist_publisher.publish(twist_msg)

			elif key == 'b':
				self.get_logger().info('AMR turn left')
				twist_msg = Twist()
				twist_msg.linear.x = 0.0
				twist_msg.angular.z = 0.01
				self.amr_twist_publisher.publish(twist_msg)

			elif key == 'n':
				self.get_logger().info('AMR turn right')
				twist_msg = Twist()
				twist_msg.linear.x = 0.0
				twist_msg.angular.z = -0.01
				self.amr_twist_publisher.publish(twist_msg)

			elif key == 'u':
				self.get_logger().info('Take photo')
				self.take_picture_execute()

			elif key == 'j':
				self.get_logger().info('Move to home position')
				self.homing_execute()

			elif key == 'k':
				self.get_logger().info('Increase speed')
				self.increase_manual_speed()

			elif key == 'l':
				self.get_logger().info('Decrease speed')
				self.decrease_manual_speed()

			elif key == 'i':
				self.get_logger().info('Show help')
				self.print_manual_mode_help()

			elif key == 'o':
				self.get_logger().info('Stop AMR')
				self.stop_amr()

			elif ord(key) == 27 if len(key) == 1 else key == '27':  # ESC key
				self.get_logger().info('ESC pressed - Disable manual mode')
				self.manual_mode_ = False
				self.set_parameters([rclpy.parameter.Parameter('manual_mode',
					rclpy.parameter.Parameter.Type.BOOL, self.manual_mode_)])
				self.stop_amr()  # 停止 AMR 移動
				self.homing_execute()  # 回到家位置

			else:
				self.get_logger().info(f'Unknown key: {key} (ASCII: {ord(key) if len(key) == 1 else "N/A"})')

		except Exception as e:
			self.get_logger().error(f'Error processing key {key}: {str(e)}')

		# 添加延時以防止過快連續操作
		time.sleep(0.1)

####################################################

	def timer_callback(self):
		pass
		# if self.tm12_feedback_.e_stop:
		# 	# 如果 TM12 處於 E-Stop 狀態，則不執行任何動作
		# 	self.cancel_callback(None)  # 取消當前目標
		# 	self.get_logger().warn('TM12 is in E-Stop state. Please release E-Stop to continue.')
		# 	return

####################################################

	def call_tm12_set_positions(self,
				   motion_type = 2,
				   positions = [0.3571, -0.5795, 0.5, -pi, 0.0, pi/4],
				   velocity = 1.5,
				   acc_time = 1.0,
				   blend_percentage = 0,
				   fine_goal = True
				   ):
		try:
			req = SetPositions.Request()
			if motion_type in [1, 2, 4]:
				req.motion_type = motion_type
			else:
				raise ValueError('Motion_type is not allowed: PTP_J = 1, PTP_T = 2, LINE_T = 4')
			req.positions = self.range_check(motion_type, positions)

			req.velocity = min(velocity, self.velocity_max_)
			req.acc_time = max(acc_time, self.acc_time_min_)
			req.blend_percentage = min(100, max(0, int(blend_percentage + 0.5)))
			req.fine_goal = fine_goal

			future = self.tm12_set_positions_client.call(req)
			return future

		except Exception as e:
			self.get_logger().error(f'設定 TM12 位置時發生錯誤: {str(e)}')
			raise

	def call_grpr2f85_set_gripper_state(self,
					   position = 0,
					   speed = 255,
					   force = 255,
					   wait_time = 0
					   ):
		try:
			req = SetGripperState.Request()
			req.position = min(255, max(0, int(position + 0.5)))
			req.speed = min(255, max(0, int(speed + 0.5)))
			req.force = min(255, max(0, int(force + 0.5)))
			req.wait_time = max(int(wait_time + 0.5), 0)

			future = self.grpr2f85_set_gripper_state_client.call(req)
			return future

		except Exception as e:
			self.get_logger().error(f'設定 2f85 夾爪狀態時發生錯誤: {str(e)}')
			raise

	def call_grpr2f85_get_gripper_status(self):
		try:
			req = GetGripperStatus.Request()

			future = self.grpr2f85_get_gripper_status_client.call(req)
			return future

		except Exception as e:
			self.get_logger().error(f'取得 2f85 夾爪狀態時發生錯誤: {str(e)}')
			raise

	def call_amr_servo_on(self):
		try:
			req = Trigger.Request()

			future = self.amr_servo_on_client.call_async(req)
			return future

		except Exception as e:
			self.get_logger().error(f'啟動 AMR 伺服時發生錯誤: {str(e)}')
			raise

	def call_amr_servo_off(self):
		try:
			req = Trigger.Request()

			future = self.amr_servo_off_client.call_async(req)
			return future

		except Exception as e:
			self.get_logger().error(f'關閉 AMR 伺服時發生錯誤: {str(e)}')
			raise

	def call_ai_inference(self, task_id: Optional[str] = None,
				text_input: Optional[str] = None) -> Optional[Infer.Response]:
		"""
		Synchronously calls the AI inference service using internally stored ROS messages.

		Parameters:
			text_input: Optional Python string for text-based inference.

		Returns:
			Optional[Infer.Response]: The service response if successful, None otherwise.
		"""
		try:
			# 為了同步調用，服務客戶端最好在與調用者相同的回調組中，
			# 或者調用者不在回調組中（例如，在 __init__ 或一次性腳本中）。
			# 如果在不同的互斥回調組中，可能會導致死鎖。
			# 如果 ai_inference_client 在 Reentrant_cb_group 中，
			# 且此函數從 Reentrant_cb_group 的回調（如 Action Server）中調用，則通常安全。

			# 檢查服務是否可用 (同步調用通常有自己的超時，但預先檢查是好的)
			if not self.ai_inference_client.wait_for_service(timeout_sec=5.0):
				self.get_logger().error('AI inference service not available before synchronous call.')
				return None

			request = Infer.Request()
			request.task_id = task_id # Set the task_id in the request

			request.text_input = text_input if text_input is not None else ""

			self.get_logger().info(f"Sending synchronous request to AI inference service '{self.ai_inference_client.srv_name}'...")

			# 同步調用服務
			# 注意：client.call() 會阻塞直到服務回應或客戶端預設的超時。
			# 您可能需要在創建客戶端時配置超時，或依賴服務本身的超時。
			response = self.ai_inference_client.call(request)

			if response is None: # 這通常表示調用本身失敗（例如，服務未運行且超時）
				self.get_logger().error(f"AI inference service call to '{self.ai_inference_client.srv_name}' failed (returned None).")
				return response, task_id

			# response 物件已包含 success, message, result_json
			return response, task_id

		except Exception as e: # 例如 rclpy.handle.InvalidHandle if node is shutting down
			self.get_logger().error(f'Error during synchronous call to AI inference service: {str(e)}', exc_info=True)
			return response, task_id

	def call_report_success2ai(self, task_id: str, grpr_status_code: int = 3) -> Optional[Success.Response]:
		"""向 AI 服務報告成功或失敗

		Args:
			task_id (str): 任務 ID
			success (int): 成功標誌，1 表示成功，0 表示失敗
			message (str): 附加訊息

		Returns:
			Optional[Success.Response]: 成功回應或 None
		"""
		try:
			if not self.report_success2ai_client.wait_for_service(timeout_sec=5.0):
				self.get_logger().error('Report success service not available.')
				return None

			req = Success.Request()
			req.task_id = task_id
			req.gripper_status_code = grpr_status_code

			future = self.report_success2ai_client.call(req)
			return future

		except Exception as e:
			self.get_logger().error(f'Error reporting success to AI service: {str(e)}')
			return None

####################################################

	def homing_callback(self, request, response):
		"""處理歸零/回到原位的服務請求

		Args:
			request (Trigger.Request): 觸發請求
			response (Trigger.Response): 服務回應

		Returns:
			Trigger.Response: 包含執行結果的回應
		"""
		try:
			self.homing_execute()
			response.success = True
			response.message = "歸零動作完成"
		except Exception as e:
			self.get_logger().error(f'歸零過程發生錯誤: {e}')
			response.success = False
			response.message = f"歸零失敗: {str(e)}"
		return response

	def homing_execute(self):
		try:
			future_tm12 = self.call_tm12_set_positions(
			motion_type=1,
			positions=self.home_in_joint_
			)
			future_gripper = self.call_grpr2f85_set_gripper_state(
			position=0
			)

		except Exception as e:
			self.get_logger().error(f'歸零過程發生錯誤: {e}')
			raise

	def take_picture_callback(self, request, response):
		"""處理拍照請求

		Args:
			request (Trigger.Request): 觸發請求
			response (Trigger.Response): 服務回應

		Returns:
			Trigger.Response: 包含執行結果的回應
		"""
		try:
			self.take_picture_execute()
			response.success = True
			response.message = "拍照完成"
		except Exception as e:
			self.get_logger().error(f'拍照過程發生錯誤: {e}')
			response.success = False
			response.message = f"拍照失敗: {str(e)}"
		return response

	def take_picture_execute(self):
		"""執行拍照的邏輯"""
		'''
		rgb->png, depth->png,npz
		'''
		try:
			if self.rgb_img_ is None or self.depth_img_ is None:
				self.get_logger().warn('RGB 或深度影像尚未準備好')
				return

			rgb = self.rgb_img_.copy()
			depth = self.depth_img_.copy()

			# 確保目錄存在
			captured_images_dir = '/workspaces/AI_Robot_ws/Hardware/src/ncku_csie_rl/tm12_amm/data/captured_images'
			rgb_dir = captured_images_dir + '/rgb'
			depth_dir = captured_images_dir + '/depth'
			os.makedirs(rgb_dir, exist_ok=True)
			os.makedirs(depth_dir, exist_ok=True)

			# 計算現有檔案數量來決定新檔案的索引
			rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith('.png')]
			depth_files = [f for f in os.listdir(depth_dir) if f.endswith('.png')]
			time.sleep(0.5)
			# 使用最大索引+1作為新檔案的索引
			rgb_index = len(rgb_files) + 1
			depth_index = len(depth_files) + 1

			# 儲存檔案
			rgb_filename = f'{rgb_dir}/rgb_{rgb_index:04d}.png'
			#depth_png_filename = f'{depth_dir}/depth_{depth_index:04d}.png'
			depth_npz_filename = f'{depth_dir}/depth_{depth_index:04d}.npz'
			cv2.imwrite(rgb_filename, rgb)
			#cv2.imwrite(depth_png_filename, depth)
			np.savez(depth_npz_filename, depth=depth)

			self.get_logger().info(f'已儲存 RGB 影像: {rgb_filename}')
			#self.get_logger().info(f'已儲存深度影像: {depth_png_filename} 和 {depth_npz_filename}')

		except Exception as e:
			self.get_logger().error(f'拍照過程發生錯誤: {e}')
			raise

####################################################
#action_clt
####################################################

	def goal_callback(self, goal_request):
		"""處理新的 goal 請求"""
		self.get_logger().info('Received goal request')
		# 在這裡可以加入 goal 的驗證邏輯
		if goal_request.task in ['Calibration', 'Verify_Calibration', 'AI_Action']:
			return GoalResponse.ACCEPT
		return GoalResponse.REJECT

	def cancel_callback(self, goal_handle):
		"""處理取消請求"""
		self.get_logger().info('Received cancel request')
		if self.tm12_feedback_.e_stop:
			self.get_logger().warn('E-Stop is active')
			return CancelResponse.ACCEPT
		self.homing_execute()
		return CancelResponse.ACCEPT

	def execute_callback(self, goal_handle: ServerGoalHandle):
		"""執行 goal 的回調函數"""
		if self.manual_mode_:
			self.get_logger().warn('Cannot execute automatic actions in manual mode')
			goal_handle.abort()
			return Dotask.Result()

		self.get_logger().info('Executing goal...')
		result = Dotask.Result()

		try:
			if goal_handle.request.task in self.action_map:
				self.action_map[goal_handle.request.task](
					goal_handle.request.scenario,
					goal_handle.request.repeat_times)
				result.ok = True
				result.result = 'Success'
			else:
				goal_handle.abort()
				result.ok = False
				result.result = 'Unknown task'
				return result

			goal_handle.succeed()
			self.get_logger().info('Goal succeeded')
			return result

		except Exception as e:
			self.get_logger().error(f'執行目標時發生錯誤: {str(e)}')
			goal_handle.abort()
			result.ok = False
			result.result = str(e)
			return result
	def calibration(self, scenario, repeat_times=1):
		"""執行校正任務"""
		try:
			self.get_logger().info('開始執行相機校正任務...')

			# 創建校正器實例
			config_path = "/workspaces/AI_Robot_ws/Hardware/src/tm12_amm/tm12_amm/config/calibration_camera_eih_001.yaml"
			calibrator = TM12Calibration(config_path)

			# 獲取校正軌跡
			trajectory = calibrator.get_trajectory()
			self.get_logger().info(f'校正軌跡包含 {len(trajectory)} 個位置')

			# 遍歷每個校正位置
			for i, pose in enumerate(trajectory):
				self.get_logger().info(f'移動到校正位置 {i+1}/{len(trajectory)}: {pose}')

				# 移動機器人到校正位置
				try:
					self.call_tm12_set_positions(
						motion_type=2,
						positions=pose,
						velocity=1.0,
						acc_time=1.0,
						blend_percentage=0,
						fine_goal=True
					)

					# 等待機器人到達位置
					self.wait_for_tm12_arrive(pose, tolerance=0.01)
					self.get_logger().info(f'機器人已到達校正位置 {i+1}')

					# 等待相機穩定
					time.sleep(2.0)

					# 獲取當前實際位置
					if self.tm12_feedback_ is None:
						raise RuntimeError("TM12 feedback not available")

					actual_pose = self.tm12_feedback_.tool_pose.copy()

					# 獲取相機影像
					if self.rgb_img_ is None or self.depth_img_ is None:
						raise RuntimeError("Camera images not available")

					rgb_img = self.rgb_img_.copy()
					depth_img = self.depth_img_.copy()

					# 添加校正數據
					success = calibrator.append_data(rgb_img, depth_img, actual_pose)
					if not success:
						self.get_logger().error(f'無法添加校正數據，位置 {i+1}')
						continue

					self.get_logger().info(f'成功收集校正數據，位置 {i+1}')

				except Exception as e:
					self.get_logger().error(f'校正位置 {i+1} 處理失敗: {e}')
					continue

			# 校正完成後回到原位
			self.get_logger().info('校正數據收集完成，回到原位...')
			self.homing_execute()

			# 執行校正計算
			self.get_logger().info('開始執行校正計算...')
			success = calibrator.execute()

			if success:
				result_path = calibrator.get_result_path()
				self.get_logger().info(f'校正成功完成！結果保存在: {result_path}')
			else:
				raise RuntimeError("校正計算失敗")

		except Exception as e:
			self.get_logger().error(f'校正過程發生錯誤: {e}')
			# 確保機器人回到安全位置
			try:
				self.homing_execute()
			except:
				pass
			raise

	def ai_action(self, scenario: str, repeat_times: int = 1):
		"""執行 AI 動作 (使用同步服務調用)

		Args:
			scenario (str): 場景名稱，可能用作 AI 的文本輸入
			repeat_times (int): 重複執行次數
		"""
		try:
			self.get_logger().info(f'開始執行 AI 動作: "{scenario}"，重複 {repeat_times} 次')
			action_succeeded_overall = True

			for i in range(repeat_times):
				current_task_id = uuid.uuid4().hex # Generate a unique task ID
				self.get_logger().info(f'開始執行 AI 動作: "{scenario}", 第 {i+1}/{repeat_times} 次, 任務 ID: {current_task_id}')

				self.call_tm12_set_positions(
				positions=self.pose_take_photo_static_
				)
				self.wait_for_tm12_arrive(self.pose_take_photo_static_)
				time.sleep(0.5)  # 等待機器人到達拍照位置
			 	# 取得當前機器人位置的變換矩陣
				T_Grpr2Base = self.get_transform_from_tm12_cartesian_pose(
			 	self.tm12_feedback_.tool_pose
			 	)

				self.get_logger().info(f'機器人已到達拍照位置，準備進行 AI 推理...')



				ai_response, task_id = self.call_ai_inference(current_task_id)

				if ai_response is None or not ai_response.success:
					self.get_logger().error(f"AI inference failed for task_id {task_id}. Scenario: '{scenario}'. Message: {ai_response.message if ai_response else 'No response'}")
					action_succeeded_overall = False
					if task_id: # Report failure if we have a task_id
						self.call_report_success2ai(task_id)
					raise RuntimeError(f"AI inference failed for task_id {task_id}. Message: {ai_response.message if ai_response else 'No response'}")

				self.get_logger().info(f"AI inference successful for task_id {task_id}. Result: {ai_response.result_json}")

				# 處理 AI 推理結果

			 	# T_obj2cam 2 T_obj2base
				T_Obj2Cam = self.get_transform_from_tm12_cartesian_pose(ai_response.result_pose)
				T_Obj2Base = self.get_transform_Obj2Base(T_Grpr2Base, T_Obj2Cam)
				pose_obj2base = self.get_tm12_cartesian_pose_from_matrix(T_Obj2Base)

				_, grpr_sc = self.pick_at(pose_obj2base, 1)# [0.3571, -0.5795, 0.2, -3.1415, 0., 0.7854]
				self.get_logger().info(f'grpr_sc: {grpr_sc}')

				self.call_report_success2ai(
					task_id=task_id,
					grpr_status_code=grpr_sc
				)

				self.place_at([0.37, -0.07, 0.05, -3.1415, 0., 0.7854])
				self.get_logger().info(f'AI 動作 "{scenario}" 第 {i+1}/{repeat_times} 次執行完畢。')
				if i < repeat_times - 1:
					time.sleep(0.5)

			if action_succeeded_overall:
				self.get_logger().info(f"AI action for scenario '{scenario}' completed successfully overall.")
			else:
				self.get_logger().error(f"AI action for scenario '{scenario}' failed or had partial failures.")

			self.homing_execute() # 假設 homing_execute 內部處理其服務調用的同步/異步

		except Exception as e:
			self.get_logger().error(f'AI 動作 "{scenario}" 執行期間發生錯誤: {e}', exc_info=True)
			try:
				self.get_logger().info("AI 動作出錯，嘗試回到原位...")
				self.homing_execute()
			except Exception as home_e:
				self.get_logger().error(f'AI 動作出錯後，回到原位時也發生錯誤: {home_e}')
			raise

####################################################

	def get_transform_from_tm12_cartesian_pose(self, tm12_pose=[0.3571, -0.5795, 0.5, -pi, 0.0, pi/4]):
		"""將 TM12 笛卡爾座標轉換為齊次變換矩陣

		Args:
			tm12_pose (list): [x, y, z, rx, ry, rz] 位置和歐拉角 (弧度)

		Returns:
			np.ndarray: 4x4 齊次變換矩陣
		"""
		try:
			# 檢查輸入
			if len(tm12_pose) != 6:
				raise ValueError("TM12 pose 必須包含 6 個元素 [x, y, z, rx, ry, rz]")

			# 從歐拉角獲取旋轉矩陣 (使用 XYZ 順序)
			rx, ry, rz = tm12_pose[3:6]

			# 分別計算各軸的旋轉矩陣
			Rx = np.array([
			[1, 0, 0],
			[0, np.cos(rx), -np.sin(rx)],
			[0, np.sin(rx), np.cos(rx)]
			])

			Ry = np.array([
			[np.cos(ry), 0, np.sin(ry)],
			[0, 1, 0],
			[-np.sin(ry), 0, np.cos(ry)]
			])

			Rz = np.array([
			[np.cos(rz), -np.sin(rz), 0],
			[np.sin(rz), np.cos(rz), 0],
			[0, 0, 1]
			])

			# 組合旋轉矩陣 (ZYX 順序)
			R = Rz @ Ry @ Rx

			# 建立齊次變換矩陣
			T = np.eye(4)
			T[:3, :3] = R
			T[:3, 3] = tm12_pose[:3]

			return T #T_Gripper2Base

		except Exception as e:
			self.get_logger().error(f'轉換 TM12 笛卡爾座標時發生錯誤: {str(e)}')
			raise

	def get_tm12_cartesian_pose_from_matrix(self, T_Gripper2Base):
		"""將齊次變換矩陣轉換為 TM12 笛卡爾座標

		Args:
			T_Gripper2Base (np.ndarray): 4x4 齊次變換矩陣

		Returns:
			list: [x, y, z, rx, ry, rz] 位置和歐拉角 (弧度)
		"""
		try:
			# 檢查輸入矩陣
			if not isinstance(T_Gripper2Base, np.ndarray) or T_Gripper2Base.shape != (4, 4):
				raise ValueError("輸入必須是 4x4 齊次變換矩陣")

			# 提取位置向量
			position = T_Gripper2Base[:3, 3]

			# 提取旋轉矩陣
			R = T_Gripper2Base[:3, :3]

			# 計算歐拉角 (使用 ZYX 順序)
			# ry = arcsin(-r31)
			ry = np.arcsin(-R[2, 0])

			# rx = arctan2(r32, r33)
			rx = np.arctan2(R[2, 1], R[2, 2])

			# rz = arctan2(r21, r11)
			rz = np.arctan2(R[1, 0], R[0, 0])

			# 組合結果
			tm12_pose = [
			position[0],  # x
			position[1],  # y
			position[2],  # z
			rx,          # 繞 X 軸旋轉
			ry,          # 繞 Y 軸旋轉
			rz           # 繞 Z 軸旋轉
			]

			return tm12_pose

		except Exception as e:
			self.get_logger().error(f'從齊次變換矩陣轉換 TM12 笛卡爾座標時發生錯誤: {str(e)}')
			raise

	def get_transform_Obj2Base(self, T_Grpr2Base, T_Obj2Cam):
		# Obj2Base = Grpr2Base Cam2Grpr Obj2Cam
		# Grpr2Base: 拍照時的位置
		# Cam2Grpr: 相機到夾爪的轉換(常數)
		# Obj2Cam: 物件到相機的轉換(從模型得到)
		return T_Grpr2Base @ (self.T_Cam2Grpr_offset_ @ (self.T_Cam2Grpr_ @ T_Obj2Cam))

	def range_check(self, motion_type, position):
		if motion_type in [2, 4]:
			position[2] = max(position[2], self.z_min_)
		return position

	def wait_for_tm12_arrive(self, target_pose, tolerance=0.001):
		"""檢查 TM12 是否到達目標位置

		Args:
			target_pose (list): 目標位置 [x, y, z, rx, ry, rz]
			tolerance (float, optional): 容許誤差. Defaults to 0.01.

		Returns:
			bool: 是否到達目標位置
		"""
		try:
			# 檢查目標位置是否為有效列表
			if len(target_pose) != 6:
				raise ValueError("目標位置必須包含 6 個元素 [x, y, z, rx, ry, rz]")

			# 檢查 TM12 回饋是否有效
			if self.tm12_feedback_ is None:
				raise ValueError("TM12 回饋為空")

			# 計算位置誤差
			while np.linalg.norm(np.array(target_pose) - np.array(self.tm12_feedback_.tool_pose)) > tolerance:
				#self.get_logger().info('等待機器人到達目標位置')
				time.sleep(0.01)


		except Exception as e:
			self.get_logger().error(f'檢查 TM12 是否到達目標位置時發生錯誤: {str(e)}')

	def pick_at(self, pose_obj2base, openning):
		PRE_GRASP_OFFSET = 0.1
		try:
			# 計算物體相對於基座的變換
			T_Obj2Base = self.get_transform_from_tm12_cartesian_pose(pose_obj2base)

			# 計算預抓取位置
			approach_vector = T_Obj2Base[:3, 2]
			T_PreGrasp2Base = T_Obj2Base.copy()
			T_PreGrasp2Base[:3, 3] -= approach_vector * PRE_GRASP_OFFSET

			# 轉換為笛卡爾座標
			pre_grasp_pose = self.get_tm12_cartesian_pose_from_matrix(T_PreGrasp2Base)
			target_pose = self.get_tm12_cartesian_pose_from_matrix(T_Obj2Base)

			# 1. 開啟夾爪並確認
			self.call_grpr2f85_set_gripper_state(position=0)

			# 2. 移動到預抓取位置
			self.call_tm12_set_positions(positions=pre_grasp_pose)

			# 3. 移動到目標位置
			self.call_tm12_set_positions(motion_type=4, positions=target_pose)

			self.wait_for_tm12_arrive(target_pose)

			self.get_logger().info('機器人已到達目標位置')
			self.call_grpr2f85_set_gripper_state(position=int(openning * 255 + 0.5), wait_time=0)

			# 5. 提起物體回到預抓取位置
			self.call_tm12_set_positions(motion_type=4, positions=pre_grasp_pose)
			resp = self.call_grpr2f85_get_gripper_status()


			return resp.ok, resp.status_code

		except Exception as e:
			self.get_logger().error(f'抓取過程發生錯誤: {str(e)}')
			# 發生錯誤時嘗試回到安全位置
			try:
				self.homing_execute()
			except:
				pass
			raise

	def place_at(self, pose_obj2base):
		PRE_GRASP_OFFSET = 0.3
		try:
			# 計算物體相對於基座的變換
			T_Obj2Base = self.get_transform_from_tm12_cartesian_pose(pose_obj2base)

			# 計算預抓取位置
			approach_vector = T_Obj2Base[:3, 2]
			T_PreGrasp2Base = T_Obj2Base.copy()
			T_PreGrasp2Base[:3, 3] -= approach_vector * PRE_GRASP_OFFSET

			# 轉換為笛卡爾座標
			pre_grasp_pose = self.get_tm12_cartesian_pose_from_matrix(T_PreGrasp2Base)
			target_pose = self.get_tm12_cartesian_pose_from_matrix(T_Obj2Base)

			# 2. 移動到預放置位置
			self.call_tm12_set_positions(positions=pre_grasp_pose)

			# 3. 移動到目標位置
			self.call_tm12_set_positions(motion_type=4, positions=target_pose)

			self.wait_for_tm12_arrive(target_pose)

			self.call_grpr2f85_set_gripper_state(position=0, wait_time=0)

			# 5. 回到預放置位置
			self.call_tm12_set_positions(motion_type=4, positions=pre_grasp_pose)

			return True

		except Exception as e:
			self.get_logger().error(f'抓取過程發生錯誤: {str(e)}')
			# 發生錯誤時嘗試回到安全位置
			try:
				self.homing_execute()
			except:
				pass
			raise

	def increase_manual_speed(self):
		"""增加手動操作速度"""
		try:
			old_speed = self.manual_speed_
			self.manual_speed_ = min(self.manual_speed_ + self.manual_speed_step_, self.manual_speed_max_)
			self.get_logger().info(f'Manual speed increased from {old_speed:.1f} to {self.manual_speed_:.1f}')

			# 更新參數
			self.set_parameters([rclpy.parameter.Parameter('manual_speed',
				rclpy.parameter.Parameter.Type.DOUBLE, self.manual_speed_)])
		except Exception as e:
			self.get_logger().error(f'Error increasing manual speed: {str(e)}')

	def decrease_manual_speed(self):
		"""減少手動操作速度"""
		try:
			old_speed = self.manual_speed_
			self.manual_speed_ = max(self.manual_speed_ - self.manual_speed_step_, self.manual_speed_min_)
			self.get_logger().info(f'Manual speed decreased from {old_speed:.1f} to {self.manual_speed_:.1f}')

			# 更新參數
			self.set_parameters([rclpy.parameter.Parameter('manual_speed',
				rclpy.parameter.Parameter.Type.DOUBLE, self.manual_speed_)])
		except Exception as e:
			self.get_logger().error(f'Error decreasing manual speed: {str(e)}')

	def toggle_manual_mode(self):
		"""切換手動模式開關"""
		try:
			self.manual_mode_ = not self.manual_mode_
			self.get_logger().info(f'Manual mode {"enabled" if self.manual_mode_ else "disabled"}')

			# 更新參數
			self.set_parameters([rclpy.parameter.Parameter('manual_mode',
				rclpy.parameter.Parameter.Type.BOOL, self.manual_mode_)])
		except Exception as e:
			self.get_logger().error(f'Error toggling manual mode: {str(e)}')

	def exit_manual_mode(self):
		"""退出手動模式"""
		try:
			self.manual_mode_ = False
			self.get_logger().info('Manual mode disabled')

			# 更新參數
			self.set_parameters([rclpy.parameter.Parameter('manual_mode',
				rclpy.parameter.Parameter.Type.BOOL, self.manual_mode_)])
		except Exception as e:
			self.get_logger().error(f'Error exiting manual mode: {str(e)}')

	def stop_amr(self):
		"""停止 AMR 移動"""
		try:
			twist_msg = Twist()
			twist_msg.linear.x = 0.0
			twist_msg.linear.y = 0.0
			twist_msg.linear.z = 0.0
			twist_msg.angular.x = 0.0
			twist_msg.angular.y = 0.0
			twist_msg.angular.z = 0.0
			self.amr_twist_publisher.publish(twist_msg)
			self.get_logger().info('AMR stopped')
		except Exception as e:
			self.get_logger().error(f'Error stopping AMR: {str(e)}')

	def print_manual_mode_help(self):
		"""打印手動模式幫助信息"""
		help_text = f"""
		Manual Control Help (Current Speed: {self.manual_speed_:.1f}):
		機械手臂控制:
		  q/a - +X/-X
		  w/s - +Y/-Y
		  e/d - +Z/-Z
		  r/f - +Roll/-Roll
		  t/g - +Pitch/-Pitch
		  y/h - +Yaw/-Yaw

		夾爪控制:
		  z - 夾爪關閉
		  x - 夾爪打開

		AMR 移動控制:
		  c - 前進
		  v - 後退
		  b - 左轉
		  n - 右轉
		  o - 停止 AMR

		速度控制:
		  k - 增加速度 (範圍: {self.manual_speed_min_:.1f} - {self.manual_speed_max_:.1f})
		  l - 減少速度

		其他功能:
		  u - 拍照
		  j - 回到原位
		  i - 顯示幫助
		  ESC - 關閉手動模式

		服務接口:
		  ros2 service call /arm/enable_manual_mode std_srvs/srv/Trigger
		  ros2 service call /arm/disable_manual_mode std_srvs/srv/Trigger
		"""
		self.get_logger().info(help_text)

####################################################
# Service callbacks for manual mode
####################################################

	def enable_manual_mode_callback(self, request, response):
		"""處理啟用手動模式的服務請求

		Args:
			request (Trigger.Request): 觸發請求
			response (Trigger.Response): 服務回應

		Returns:
			Trigger.Response: 包含執行結果的回應
		"""
		try:
			self.manual_mode_ = True
			self.get_logger().info('手動模式已啟用')

			# 更新參數
			self.set_parameters([rclpy.parameter.Parameter('manual_mode',
				rclpy.parameter.Parameter.Type.BOOL, self.manual_mode_)])

			# 顯示幫助信息
			self.print_manual_mode_help()

			response.success = True
			response.message = "手動模式已啟用"
		except Exception as e:
			self.get_logger().error(f'啟用手動模式過程發生錯誤: {e}')
			response.success = False
			response.message = f"啟用手動模式失敗: {str(e)}"
		return response

	def disable_manual_mode_callback(self, request, response):
		"""處理停用手動模式的服務請求

		Args:
			request (Trigger.Request): 觸發請求
			response (Trigger.Response): 服務回應

		Returns:
			Trigger.Response: 包含執行結果的回應
		"""
		try:
			self.manual_mode_ = False
			self.get_logger().info('手動模式已停用')

			# 更新參數
			self.set_parameters([rclpy.parameter.Parameter('manual_mode',
				rclpy.parameter.Parameter.Type.BOOL, self.manual_mode_)])

			# 停止 AMR 移動
			self.stop_amr()

			response.success = True
			response.message = "手動模式已停用"
		except Exception as e:
			self.get_logger().error(f'停用手動模式過程發生錯誤: {e}')
			response.success = False
			response.message = f"停用手動模式失敗: {str(e)}"
		return response
####################################################
def main(args=None):
	rclpy.init(args=args)
	executor = MultiThreadedExecutor()

	node = TM12_AMM_ROS2_Node()
	executor.add_node(node)

	try:
		executor.spin()
	except KeyboardInterrupt:
		pass
	except Exception as exception:
		raise exception
	finally:
		node.destroy_node()
		if rclpy.ok():
			rclpy.shutdown()

if __name__ == '__main__':
	main()

