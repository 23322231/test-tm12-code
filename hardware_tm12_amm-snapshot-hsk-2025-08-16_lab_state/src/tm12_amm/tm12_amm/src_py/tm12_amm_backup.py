#!/usr/bin/env python3

import sys
import os
import numpy as np                                     # Python数值计算库
import cv2                                             # Opencv图像处理库
import time
import threading
import math
from math import pi


import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionServer, \
				CancelResponse, \
				GoalResponse

from cv_bridge import CvBridge                         # ROS与OpenCV图像转换类

from std_msgs.msg import Header
from sensor_msgs.msg import Image                      # 图像消息类型
from ai_interfaces.action import Simple, \
				Calibration

from tm_msgs.srv import SetPositions
from grpr2f85_ifaces.srv import SetGripperState #, \
				#Reset, \
				#GetGripperStatus
from realsense2_camera_msgs.msg import RGBD

class Ai_controller(Node):
	def __init__(self, name='ai_controller'):
		super().__init__(name)

		self.function_map = {
			"homing": self.homing,
			"calibration": self.calibration,
			"look_and_move": self.look_and_move,
			"simple_demo": self.simple_demo
		}

		self.tm12_default_setting = {
			"z_min": 0.02, # m
			"velocity_max": 2, # rad/s
			"acc_time_min": 0.2, # s
			"home_in_joint": [-pi/4, 0., pi/2, 0., pi/2, 0.],
			"home_in_cartesian": [0.3571, -0.5795, 0.5, -pi, 0., pi/4],
			#"AMR_body":
		}

		self.cv_bridge = CvBridge()

		self.realsens_data = RGBD() #

		self._action_server = ActionServer(
			self,
			Simple,
			'ai_action',
			execute_callback=self.execute_callback,
			goal_callback=self.goal_callback	,
			cancel_callback=self.cancel_callback
			)
	
		self._calibration_server = ActionServer(
			self,
			Calibration,
			'calibration',
			execute_callback=self.calibration_callback,
			goal_callback=self.goal_callback	,
			cancel_callback=self.cancel_callback
			)

	# Create clients for both AddTwoInts and MultiplyTwoInts services
		self.tm12_client = self.create_client(
			SetPositions, 
			'set_position'
			)

		self.gripper2f85_client = self.create_client(
			SetGripperState, 
			'set_gripper_state'
			)
		
		self.rgb_realsense_subscriber = self.create_subscription(
			Image, 
			'/camera/color/image_raw', 
			self.rgb_realsense_callback,
			10
			)

		self.depth_realsense_subscriber = self.create_subscription(
			Image, 
			'/camera/depth/image_raw', 
			self.depth_realsense_callback,
			10
			)

		while not self.tm12_client.wait_for_service(timeout_sec=1.0):
			self.get_logger().info('Waiting for TM12 service ...')		
		while not self.gripper2f85_client.wait_for_service(timeout_sec=1.0):
			self.get_logger().info('Waiting for 2f85 service ...')

	def goal_callback(self, goal_request):
		"""Accept all goals for simplicity."""
		self.get_logger().info('Received goal request.')
		return GoalResponse.ACCEPT

	def cancel_callback(self, goal_handle):
		"""Allow cancellations."""
		self.get_logger().info('Received cancel request.')
		return CancelResponse.ACCEPT

	def range_chack(self, motion_type, positions):
		if motion_type==2:
			positions[2] = max(positions[2],self.tm12_default_setting["z_min"])
			return positions
		
	async def call_tm_driver_set_position(self, motion_type=2, 
					positions=[0.3571, -0.5795, 0.5, -pi, 0., pi/4], velocity=2, acc_time=0.2, 
					blend_percentage=10, fine_goal = True
					): #default as home position
		req = SetPositions.Request()
		if motion_type in [1,2,4]:
			req.motion_type = motion_type
		else:
			raise ValueError('Motion_type is not allowed: PTP_J = 1, PTP_T = 2, LINE_T = 4')
		req.positions = self.range_chack(motion_type = motion_type, positions = positions) # need a function to check the envelope
		req.velocity = min(velocity,self.tm12_default_setting["velocity_max"])
		req.acc_time = max(acc_time,self.tm12_default_setting["acc_time_min"])
		req.blend_percentage = min(100,max(0,int(blend_percentage+0.5)))
		req.fine_goal = fine_goal
		
		self.get_logger().info(f'Calling tm_driver set_position service')
		future = self.tm12_client.call_async(req)
		await future
		if future.result() is None:
			raise RuntimeError('tm_driver_set_position call failed')
		return future.result()

	async def call_grpr2f85_set(self, position=0, speed=255, force=255, wait_time=0):
		req = SetGripperState.Request()
		req.position = min(255,max(0,int(position+0.5)))
		req.speed = min(255,max(0,int(speed+0.5)))
		req.force = min(255,max(0,int(force+0.5)))
		req.wait_time = max(int(wait_time+0.5),0)
		
		self.get_logger().info(f'Calling my_grpr2f85_set service')
		future = self.gripper2f85_client.call_async(req)
		await future
		if future.result() is None:
			raise RuntimeError('grpr2f85_set call failed')
		return future.result()

##########################################################################################################

	async def homing(self, goal_handle):
		act_feedback = Simple.Feedback()
		try:
			result = await self.call_tm_driver_set_position(
				motion_type=1,
				positions = self.tm12_default_setting["home_in_joint"]
			)
			result = await self.call_grpr2f85_set(
				positions = 0
			)			
			act_feedback.current_state = f'Homing'
			goal_handle.publish_feedback(act_feedback)
			self.get_logger.info(act_feedback.current_state)
		except Exception as e:
			self.get_logger().error(f'Error during homing: {e}')
			raise
		return result

#TODO modify measure, mmanipulate, calibration, simple_demo

	async def measure(self, goal_handle, measurement_traj):
		act_feedback = Simple.Feedback()
		try:
			for state in measurement_traj:
				result = await self.call_tm_driver_set_position(
					positions = state
				)
				act_feedback.current_state = f'Take picture at Pose: {state}'
				# take picture
				goal_handle.publish_feedback(act_feedback)
				self.get_logger.info(act_feedback.current_state)
		except Exception as e:
			self.get_logger().error(f'Error during measurement: {e}')
			raise
		return result

	async def manipulate(self, goal_handle, manipulation_traj):
		act_feedback = Simple.Feedback()
		try:
			for state in manipulation_traj:
				result_tm12 = await self.call_tm_driver_set_position(
					positions = state
				)
				result_2f85 = await self.call_grpr2f85_set(
				    positions = state[6]
				)
				act_feedback.current_state = f'Pose: {state}, Gripper: {result_2f85.result}'
				goal_handle.publish_feedback(act_feedback)
				result = [result_tm12, result_2f85]
				self.get_logger.info(act_feedback.current_state)
		except Exception as e:
			self.get_logger().error(f'Error during manipulation: {e}')
			raise
		return result

	async def calibration():
		return True
	
	async def look_and_move():
		return True

	async def simple_demo():
		return True

##########################################################################################################

	async def execute_callback(self, goal_handle):
		self.get_logger().info('Executing goal...')
		input_string = goal_handle.request.task
		
		if input_string in self.function_map:
			self.function_map[input_string]()
		else:
			goal_handle.abort()
			return Simple.Result()  # 返回空結果
		

		#self.algo = simple_algorithm(task='fixed', scenario='straight_up_down')
		##TODO: algo = algo_switcher()
		##TODO: can be repeat

		#t_0 = self.get_clock().now()
		## measurement
		#try:
		#	result = await self.measure(goal_handle, self.algo.get_sample_trajectory())
		#except Exception as e:
		#	goal_handle.abort()
		#	self.get_logger().error(f'Error in measurement: {e}')
		#	return Simple.Result()
		#t_1 = self.get_clock().now()
		## inference
		#nference_result = self.algo.forward()
		#t_2 = self.get_clock().now()
		## action
		#try:
		#	result = await self.manipulate(goal_handle, inference_result)
		#except Exception as e:
		#	goal_handle.abort()
		#	self.get_logger().error(f'Error in manipulation: {e}')
		#	return Simple.Result()
		#final_result = Simple.Result()
		#final_result.result = result
		
		
		goal_handle.succeed()
		t_3 = self.get_clock().now()
		self.get_logger().info(f'Goal succeeded with result: {final_result.result_data}')
		return final_result

	async def calibration_callback(self, goal_handle):
		self.get_logger().info('Executing calibration...')

def main(args=None):
	rclpy.init(args=args)
	executor = MultiThreadedExecutor()

	node = Ai_controller()
	executor.add_node(node)

	try:
		executor.spin()
	except KeyboardInterrupt:
		pass
	except Exception as exception:  
		raise exception
	finally:
		node.destroy_node()
		rclpy.shutdown()

if __name__ == '__main__':
	main()

