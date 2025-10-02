#!/usr/bin/env python3

import sys
import os
import numpy as np
import cv2
import time
import threading
import math
from math import pi
from typing import TypedDict, Optional, Dict, Any, List, Callable
import numpy.typing as npt
import signal
import logging
import asyncio
import uuid
import json
from dataclasses import dataclass
from enum import Enum
from scipy.spatial.transform import Rotation

import rclpy
import rclpy.callback_groups
import rclpy.parameter
from rclpy.task import Future
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle

import ros2_numpy as rnp
from cv_bridge import CvBridge

from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from std_msgs.msg import Header, Char, String
from geometry_msgs.msg import Twist, Pose, Quaternion
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from realsense2_camera_msgs.msg import RGBD
from tm_msgs.msg import FeedbackState

from std_srvs.srv import Trigger
from tm_msgs.srv import SetPositions
from grpr2f85_ifaces.srv import SetGripperState, GetGripperStatus
from tm12_amm_interfaces.action import Dotask, Calibration
from ai_interfaces.srv import Infer, Success

from calibration import TM12Calibration
import tm12_amm

def _execute_ai_action_auto_decide(TM12AMMNode, scenario):
    try:
        TM12AMMNode.get_logger().info(f'開始執行 AI 動作: "{scenario}"，自動偵測模式')
        counter = 0
        while True:
            if TM12AMMNode.tm12_feedback and TM12AMMNode.tm12_feedback.robot_error:
                TM12AMMNode.get_logger().warn('檢測到手臂錯誤，AI動作中止')
                raise Exception('Robot error triggered')
            task_id = uuid.uuid4().hex
            TM12AMMNode.get_logger().info(f'執行 AI 動作第 {counter+1} 次，任務 ID: {task_id}')

            # 移動到拍照位置

            TM12AMMNode.robot_controller.move_to_pose(TM12AMMNode.pose_take_photo_static)
            TM12AMMNode.get_logger().info(f'debug1')
            TM12AMMNode.robot_controller.wait_for_arrival(TM12AMMNode.pose_take_photo_static)
            time.sleep(1)
            # 獲取機器人當前位置
            gripper_pose = TM12AMMNode.tm12_feedback.tool_pose
            TM12AMMNode.get_logger().info(f'debug2')
            # 調用 AI 推理
            ai_response = TM12AMMNode._call_ai_inference(task_id)
            TM12AMMNode.get_logger().info(f'debug3')
            if not ai_response.detection_found:
                break

            if ai_response is None or not ai_response.success:
                TM12AMMNode.get_logger().error(f'AI 推理失敗，任務 ID: {task_id}')
                if task_id:
                    TM12AMMNode._report_success_to_ai(task_id, 0)
                continue
            
            # 計算物體在基座中的位置
            object_pose_in_base = tm12_amm.RobotPose.get_object_pose_in_base(
                gripper_pose, ai_response.result_pose,
                TM12AMMNode.camera_config, TM12AMMNode.hand_eye_config
            )

            # 執行抓取
            success, status_code = TM12AMMNode._pick_at_pose(object_pose_in_base, 1.0)

            if success:
                TM12AMMNode.get_logger().info(f'抓取成功，狀態碼: {status_code}')
                TM12AMMNode._report_success_to_ai(task_id, status_code)
                # 是奎的位置
                # Aruco


                # 放置物體
                TM12AMMNode._place_at_pose(TM12AMMNode.pick_place_config.default_place_pose)
            else:
                TM12AMMNode.get_logger().error(f'抓取失敗，狀態碼: {status_code}')
                TM12AMMNode._report_success_to_ai(task_id, status_code)

            counter += 1
            time.sleep(0.5)
        return
    except Exception as e:
        TM12AMMNode.get_logger().error(f'AI動作失敗: {e}')
        TM12AMMNode.robot_controller.move_to_home()
        raise
    return
