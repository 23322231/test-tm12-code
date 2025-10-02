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

# import workspace_c23322231
# import workspace_shihkuei

#TODO:
#1. 把所有座標轉換以 tf 管理

#ros2 action send_goal /robot/arm/ai_action tm12_amm_interfaces/action/Dotask "{task: 'AI_Action', scenario: '', repeat_times: 1}"
#ros2 action send_goal /robot/arm/ai_action tm12_amm_interfaces/action/Dotask "{task: 'Calibration', scenario: '', repeat_times: 0}"

class MotionType(Enum):
    """運動類型枚舉"""
    PTP_J = 1  # 關節點對點運動
    PTP_T = 2  # 笛卡爾點對點運動
    LINE_T = 4  # 直線運動


class GripperState(Enum):
    """夾爪狀態枚舉"""
    OPEN = 0
    CLOSED = 255


@dataclass
class FilePaths:
    """文件路徑配置"""
    captured_images_dir: str = '/home/robotics/hardware_tm12_amm/src/tm12_amm/tm12_amm/data/captured_images'
    calibration_config_path: str = '/home/robotics/hardware_tm12_amm/src/tm12_amm/tm12_amm/config/calibration_camera_eih_001.yaml'

    @property
    def rgb_dir(self) -> str:
        return os.path.join(self.captured_images_dir, 'rgb')

    @property
    def depth_dir(self) -> str:
        return os.path.join(self.captured_images_dir, 'depth')


@dataclass
class ServiceTimeouts:
    """服務超時配置"""
    ai_inference_timeout: float = 5.0
    report_success_timeout: float = 5.0
    general_service_timeout: float = 5.0


@dataclass
class RobotPose:
    """機器人姿態數據類 - 整合座標轉換功能"""
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z, self.rx, self.ry, self.rz]

    @classmethod
    def from_list(cls, pose_list: List[float]) -> 'RobotPose':
        if len(pose_list) != 6:
            raise ValueError("姿態列表必須包含6個元素")
        return cls(*pose_list)

    def to_matrix(self) -> np.ndarray:
        """將機器人姿態轉換為齊次變換矩陣"""
        x, y, z, rx, ry, rz = self.to_list()
        # 使用 scipy.spatial.transform.Rotation 建立旋轉矩陣
        R = Rotation.from_euler('xyz', [rx, ry, rz]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    @classmethod
    def from_matrix(cls, transform_matrix: np.ndarray) -> 'RobotPose':
        """從齊次變換矩陣創建機器人姿態"""
        if transform_matrix.shape != (4, 4):
            raise ValueError("輸入必須是4x4齊次變換矩陣")
        position = transform_matrix[:3, 3]
        R = transform_matrix[:3, :3]
        # 使用 scipy.spatial.transform.Rotation 反推歐拉角
        rx, ry, rz = Rotation.from_matrix(R).as_euler('xyz')
        return cls(position[0], position[1], position[2], rx, ry, rz)

    @staticmethod
    def compute_object_pose_in_base(gripper_pose: List[float],
                                  object_pose_in_camera: List[float],
                                  T_cam2gripper: np.ndarray,
                                  T_cam2gripper_offset: np.ndarray) -> List[float]:
        """計算物體在基座座標系中的姿態 - 靜態方法"""
        gripper_pose_obj = RobotPose.from_list(gripper_pose)
        object_pose_obj = RobotPose.from_list(object_pose_in_camera)

        T_gripper2base = gripper_pose_obj.to_matrix()
        T_object2camera = object_pose_obj.to_matrix()

        T_object2base = (T_gripper2base @
                        T_cam2gripper_offset @
                        T_cam2gripper @
                        T_object2camera)

        return RobotPose.from_matrix(T_object2base).to_list()

    @staticmethod
    def create_transform_helper(camera_config: 'CameraConfig', hand_eye_config: 'HandEyeConfig'):
        """創建座標轉換輔助器 - 返回變換矩陣"""
        T_cam2gripper = np.eye(4)
        T_cam2gripper[:3, :3] = hand_eye_config.rotation.reshape(3, 3)
        T_cam2gripper[:3, 3] = hand_eye_config.translation

        T_cam2gripper_offset = np.eye(4)
        T_cam2gripper_offset[:3, :3] = hand_eye_config.offset_rotation.reshape(3, 3)
        T_cam2gripper_offset[:3, 3] = hand_eye_config.offset_translation

        return T_cam2gripper, T_cam2gripper_offset, camera_config.matrix, camera_config.distortion

    @staticmethod
    def get_object_pose_in_base(gripper_pose: List[float],
                               object_pose_in_camera: List[float],
                               camera_config: 'CameraConfig',
                               hand_eye_config: 'HandEyeConfig') -> List[float]:
        """計算物體在基座座標系中的姿態 - 整合版本"""
        T_cam2gripper, T_cam2gripper_offset, _, _ = RobotPose.create_transform_helper(
            camera_config, hand_eye_config
        )

        return RobotPose.compute_object_pose_in_base(
            gripper_pose, object_pose_in_camera,
            T_cam2gripper, T_cam2gripper_offset
        )


@dataclass
class ManualControlConfig:
    """手動控制配置"""
    speed: float = 200.0
    speed_min: float = 50.0
    speed_max: float = 1000.0
    speed_step: float = 100.0
    enabled: bool = False


@dataclass
class RobotConfig:
    """機器人配置參數"""
    z_min: float = 0.035
    velocity_max: float = 2.0
    acc_time_min: float = 0.2
    home_joint_pose: List[float] = None

    def __post_init__(self):
        if self.home_joint_pose is None:
            self.home_joint_pose = [-pi/4, 0.0, pi/2, 0.0, pi/2, 0.0]


@dataclass
class PickPlaceConfig:
    """抓取放置配置參數"""
    pre_grasp_offset: float = 0.1
    pre_place_offset: float = 0.3
    default_place_pose: List[float] = None
    arrival_tolerance: float = 0.001
    stabilize_time: float = 2.0

    def __post_init__(self):
        if self.default_place_pose is None:
            self.default_place_pose = [0.5, 0.7, 0.10, -3.1415, 0., 0.7854]


@dataclass
class CameraConfig:
    """相機配置參數"""
    matrix: np.ndarray
    distortion: np.ndarray

    def __post_init__(self):
        if not isinstance(self.matrix, np.ndarray):
            self.matrix = np.array(self.matrix, dtype=np.float64)
        if not isinstance(self.distortion, np.ndarray):
            self.distortion = np.array(self.distortion, dtype=np.float64)
        #print(f"CameraConfig 初始化: matrix={self.matrix}, distortion={self.distortion}")

@dataclass
class HandEyeConfig:
    """手眼標定配置參數"""
    rotation: np.ndarray
    translation: np.ndarray
    offset_rotation: np.ndarray
    offset_translation: np.ndarray

    def __post_init__(self):
        if not isinstance(self.rotation, np.ndarray):
            self.rotation = np.array(self.rotation, dtype=np.float64)
        if not isinstance(self.translation, np.ndarray):
            self.translation = np.array(self.translation, dtype=np.float64)
        if not isinstance(self.offset_rotation, np.ndarray):
            self.offset_rotation = np.array(self.offset_rotation, dtype=np.float64)
        if not isinstance(self.offset_translation, np.ndarray):
            self.offset_translation = np.array(self.offset_translation, dtype=np.float64)


class ServiceManager:
    """服務管理器 - 統一管理所有服務客戶端"""

    def __init__(self, node: Node, callback_group):
        self.node = node
        self.callback_group = callback_group
        self._setup_clients()

    def _setup_clients(self):
        """設置所有服務客戶端"""
        # 夾爪服務
        self.gripper_set_state = self.node.create_client(
            SetGripperState, 'arm/gripper/grpr2f85_driver/set_gripper_state',
            callback_group=self.callback_group
        )
        self.gripper_get_status = self.node.create_client(
            GetGripperStatus, 'arm/gripper/grpr2f85_driver/get_gripper_status',
            callback_group=self.callback_group
        )

        # AMR 服務
        self.amr_servo_on = self.node.create_client(
            Trigger, 'amr/iamech/iamech_driver/servo_on',
            callback_group=self.callback_group
        )
        self.amr_servo_off = self.node.create_client(
            Trigger, 'amr/iamech/iamech_driver/servo_off',
            callback_group=self.callback_group
        )

        # TM12 服務
        self.tm12_set_positions = self.node.create_client(
            SetPositions, 'arm/tm12/tm_driver/set_positions',
            callback_group=self.callback_group
        )

        # AI 服務
        self.ai_inference = self.node.create_client(
            Infer, 'ai_inference', callback_group=self.callback_group
        )
        self.report_success = self.node.create_client(
            Success, 'report_success', callback_group=self.callback_group
        )


class RobotController:
    """機器人控制器"""

    def __init__(self, node: Node, service_manager: ServiceManager,
                 camera_config: CameraConfig, hand_eye_config: HandEyeConfig,
                 robot_config: RobotConfig):
        self.node = node
        self.service_manager = service_manager
        self.camera_config = camera_config
        self.hand_eye_config = hand_eye_config
        self.config = robot_config
        self.current_feedback = None

    def set_feedback(self, feedback: FeedbackState):
        """設置機器人反饋"""
        self.current_feedback = feedback

    def move_to_pose(self, pose: List[float], motion_type: MotionType = MotionType.PTP_T,
                     velocity: float = 1.5, acc_time: float = 1.0,
                     blend_percentage: int = 0, fine_goal: bool = True):
        """移動到指定姿態"""
        try:
            request = SetPositions.Request()
            request.motion_type = motion_type.value
            request.positions = self._range_check(motion_type, pose)
            request.velocity = min(velocity, self.config.velocity_max)
            request.acc_time = max(acc_time, self.config.acc_time_min)
            request.blend_percentage = min(100, max(0, blend_percentage))
            request.fine_goal = fine_goal

            return self.service_manager.tm12_set_positions.call(request)
        except Exception as e:
            self.node.get_logger().error(f'移動機器人時發生錯誤: {e}')
            raise

    def move_to_home(self):
        """回到家位置"""
        try:
            return self.move_to_pose(self.config.home_joint_pose, MotionType.PTP_J)
        except Exception as e:
            self.node.get_logger().error(f'回到家位置時發生錯誤: {e}')
            raise

    def wait_for_arrival(self, target_pose: List[float], tolerance: float = None, timeout: float = 30.0):
        """等待機器人到達目標位置"""
        if tolerance is None:
            tolerance = self.config.arrival_tolerance if hasattr(self.config, 'arrival_tolerance') else 0.001

        if len(target_pose) != 6:
            raise ValueError("目標位置必須包含6個元素")

        if self.current_feedback is None:
            raise ValueError("機器人反饋為空")

        start_time = time.time()

        # 先檢查是否已經在目標位置（只計算 XYZ 位置距離）
        target_xyz = np.array(target_pose[:3])
        current_xyz = np.array(self.current_feedback.tool_pose[:3])
        current_distance = np.linalg.norm(target_xyz - current_xyz)
        if current_distance <= tolerance:
            self.node.get_logger().info(f"機器人已在目標位置 (XYZ距離: {current_distance:.6f})")
            return

        self.node.get_logger().info(f"等待機器人到達目標位置 (當前XYZ距離: {current_distance:.6f}, 容差: {tolerance})")

        while True:
            target_xyz = np.array(target_pose[:3])
            current_xyz = np.array(self.current_feedback.tool_pose[:3])
            current_distance = np.linalg.norm(target_xyz - current_xyz)

            if current_distance <= tolerance:
                break

            # 檢查超時
            if time.time() - start_time > timeout:
                raise Exception(f"等待到達目標位置超時 ({timeout}秒)，當前XYZ距離: {current_distance:.6f}")

            time.sleep(0.01)

        final_distance = np.linalg.norm(target_xyz - current_xyz)
        self.node.get_logger().info(f"機器人已到達目標位置 (最終XYZ距離: {final_distance:.6f})")

    def _range_check(self, motion_type: MotionType, position: List[float]) -> List[float]:
        """檢查位置範圍"""
        if motion_type in [MotionType.PTP_T, MotionType.LINE_T]:
            position[2] = max(position[2], self.config.z_min)
        return position


class GripperController:
    """夾爪控制器"""

    def __init__(self, node: Node, service_manager: ServiceManager):
        self.node = node
        self.service_manager = service_manager

    def set_gripper_state(self, position: int = 0, speed: int = 255,
                         force: int = 255, wait_time: int = 0):
        """設置夾爪狀態"""
        try:
            request = SetGripperState.Request()
            request.position = min(255, max(0, position))
            request.speed = min(255, max(0, speed))
            request.force = min(255, max(0, force))
            request.wait_time = max(0, wait_time)

            return self.service_manager.gripper_set_state.call(request)
        except Exception as e:
            self.node.get_logger().error(f'設置夾爪狀態時發生錯誤: {e}')
            raise

    def get_gripper_status(self):
        """獲取夾爪狀態"""
        try:
            request = GetGripperStatus.Request()
            return self.service_manager.gripper_get_status.call(request)
        except Exception as e:
            self.node.get_logger().error(f'獲取夾爪狀態時發生錯誤: {e}')
            raise

    def open_gripper(self):
        """打開夾爪"""
        return self.set_gripper_state(GripperState.OPEN.value)

    def close_gripper(self):
        """關閉夾爪"""
        return self.set_gripper_state(GripperState.CLOSED.value)


class ManualController:
    """手動控制器"""

    def __init__(self, node: Node, robot_controller: RobotController,
                 gripper_controller: GripperController, config: ManualControlConfig):
        self.node = node
        self.robot_controller = robot_controller
        self.gripper_controller = gripper_controller
        self.config = config
        self._setup_key_mappings()

    def _setup_key_mappings(self):
        """設置按鍵映射"""
        self.key_mappings = {
            'q': self._move_plus_x_diagonal,
            'a': self._move_minus_x_diagonal,
            'w': self._move_plus_y_diagonal,
            's': self._move_minus_y_diagonal,
            'e': self._move_plus_z,
            'd': self._move_minus_z,
            'r': self._rotate_plus_roll,
            'f': self._rotate_minus_roll,
            't': self._rotate_plus_pitch,
            'g': self._rotate_minus_pitch,
            'y': self._rotate_plus_yaw,
            'h': self._rotate_minus_yaw,
            'z': self._close_gripper,
            'x': self._open_gripper,
            'k': self._increase_speed,
            'l': self._decrease_speed,
            'u': self._take_picture,
            'j': self._go_home,
        }

    def process_key(self, key: str):
        """處理按鍵輸入"""
        if not self.config.enabled:
            self.node.get_logger().warn('手動模式未啟用')
            return

        if self.robot_controller.current_feedback is None:
            self.node.get_logger().warn('機器人反饋不可用')
            return

        try:
            if key in self.key_mappings:
                self.key_mappings[key]()
            elif ord(key) == 27:  # ESC key
                self._exit_manual_mode()
            else:
                self.node.get_logger().info(f'未知按鍵: {key}')
        except Exception as e:
            self.node.get_logger().error(f'處理按鍵 {key} 時發生錯誤: {e}')

    def _get_current_pose(self) -> List[float]:
        """獲取當前機器人姿態"""
        return list(self.robot_controller.current_feedback.tool_pose)

    def _move_plus_x_diagonal(self):
        pose = self._get_current_pose()
        pose[0] += self.config.speed * 0.001 * np.cos(-pi/4)
        pose[1] += self.config.speed * 0.001 * np.sin(-pi/4)
        self.robot_controller.move_to_pose(pose)

    def _move_minus_x_diagonal(self):
        pose = self._get_current_pose()
        pose[0] -= self.config.speed * 0.001 * np.cos(-pi/4)
        pose[1] -= self.config.speed * 0.001 * np.sin(-pi/4)
        self.robot_controller.move_to_pose(pose)

    def _move_plus_y_diagonal(self):
        pose = self._get_current_pose()
        pose[0] += self.config.speed * 0.001 * (-np.sin(-pi/4))
        pose[1] += self.config.speed * 0.001 * np.cos(-pi/4)
        self.robot_controller.move_to_pose(pose)

    def _move_minus_y_diagonal(self):
        pose = self._get_current_pose()
        pose[0] -= self.config.speed * 0.001 * (-np.sin(-pi/4))
        pose[1] -= self.config.speed * 0.001 * np.cos(-pi/4)
        self.robot_controller.move_to_pose(pose)

    def _move_plus_z(self):
        pose = self._get_current_pose()
        pose[2] += self.config.speed * 0.001
        self.robot_controller.move_to_pose(pose)

    def _move_minus_z(self):
        pose = self._get_current_pose()
        pose[2] -= self.config.speed * 0.001
        self.robot_controller.move_to_pose(pose)

    def _rotate_plus_roll(self):
        pose = self._get_current_pose()
        pose[3] += self.config.speed * pi / 1800
        self.robot_controller.move_to_pose(pose)

    def _rotate_minus_roll(self):
        pose = self._get_current_pose()
        pose[3] -= self.config.speed * pi / 1800
        self.robot_controller.move_to_pose(pose)

    def _rotate_plus_pitch(self):
        pose = self._get_current_pose()
        pose[4] += self.config.speed * pi / 1800
        self.robot_controller.move_to_pose(pose)

    def _rotate_minus_pitch(self):
        pose = self._get_current_pose()
        pose[4] -= self.config.speed * pi / 1800
        self.robot_controller.move_to_pose(pose)

    def _rotate_plus_yaw(self):
        pose = self._get_current_pose()
        pose[5] += self.config.speed * pi / 1800
        self.robot_controller.move_to_pose(pose)

    def _rotate_minus_yaw(self):
        pose = self._get_current_pose()
        pose[5] -= self.config.speed * pi / 1800
        self.robot_controller.move_to_pose(pose)

    def _close_gripper(self):
        self.gripper_controller.close_gripper()

    def _open_gripper(self):
        self.gripper_controller.open_gripper()

    def _increase_speed(self):
        old_speed = self.config.speed
        self.config.speed = min(self.config.speed + self.config.speed_step,
                               self.config.speed_max)
        self.node.get_logger().info(f'速度從 {old_speed:.1f} 增加到 {self.config.speed:.1f}')

    def _decrease_speed(self):
        old_speed = self.config.speed
        self.config.speed = max(self.config.speed - self.config.speed_step,
                               self.config.speed_min)
        self.node.get_logger().info(f'速度從 {old_speed:.1f} 降低到 {self.config.speed:.1f}')


    def _take_picture(self):
        # 這裡需要調用主節點的拍照功能
        pass

    def _go_home(self):
        self.robot_controller.move_to_home()

    def _exit_manual_mode(self):
        self.config.enabled = False
        self.node.get_logger().info('手動模式已退出')


class TM12AMMNode(Node):
    """TM12 AMM 主節點 - 重構版本"""

    def __init__(self, name: str = 'tm12_amm'):
        super().__init__(name)
        self.get_logger().info('TM12 AMM Node (重構版本) 初始化中...')

        # 初始化基本組件
        self.cv_bridge = CvBridge()
        self._setup_callback_groups()
        self._setup_parameters()
        self._setup_data_storage()
        self._setup_components()
        self._setup_ros_interfaces()

        # 設置定時器和回調
        self.timer = self.create_timer(1.0, self._timer_callback,
                                      callback_group=self.reentrant_cb_group)
        self.parameter_callback_handle = self.add_on_set_parameters_callback(
            self._parameter_callback)

        self.get_logger().info('TM12 AMM Node 初始化完成')

    def _setup_callback_groups(self):
        """設置回調組"""
        self.mutually_exclusive_cb_group = MutuallyExclusiveCallbackGroup()
        self.reentrant_cb_group = ReentrantCallbackGroup()

    def _setup_parameters(self):
        """設置參數"""
        read_only = ParameterDescriptor(read_only=True)

        # 機器人配置參數
        self.robot_config = RobotConfig(
            z_min=self.declare_parameter('z_min', 0.035, read_only).value,
            velocity_max=self.declare_parameter('velocity_max', 2.0, read_only).value,
            acc_time_min=self.declare_parameter('acc_time_min', 0.2, read_only).value,
            home_joint_pose=self.declare_parameter(
                'home_in_joint', [-pi/4, 0.0, pi/2, 0.0, pi/2, 0.0], read_only).value
        )

        # 相機配置參數（改為由 topic 獲取，原本 config 讀取註解）
        # camera_matrix_list = self.declare_parameter('camera.matrix', [
        #     1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], read_only).value
        # dist_coeffs_list = self.declare_parameter('camera.distortion',
        #                                      [0.0, 0.0, 0.0, 0.0, 0.0], read_only).value
        # self.camera_config = CameraConfig(
        #     matrix=np.array(camera_matrix_list, dtype=np.float64).reshape(3, 3),
        #     distortion=np.array(dist_coeffs_list, dtype=np.float64)
        # )
        self.camera_config = None  # 等待 CameraInfo topic 訂閱
        self._camera_info_sub = self.create_subscription(
            CameraInfo, "arm/camera/realsense_camera/color/camera_info",
            self._camera_info_callback, 1
        )

        # 手眼標定配置參數
        R_c2g_list = self.declare_parameter('hand_eye.rotation', [
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], read_only).value
        t_c2g_list = self.declare_parameter('hand_eye.translation',
                                           [0.0, 0.0, 0.0], read_only).value
        R_c2g_offset_list = self.declare_parameter('hand_eye_offset.rotation', [
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], read_only).value
        t_c2g_offset_list = self.declare_parameter('hand_eye_offset.translation',
                                                  [0.0, 0.0, 0.0], read_only).value

        self.hand_eye_config = HandEyeConfig(
            rotation=np.array(R_c2g_list, dtype=np.float64),
            translation=np.array(t_c2g_list, dtype=np.float64),
            offset_rotation=np.array(R_c2g_offset_list, dtype=np.float64),
            offset_translation=np.array(t_c2g_offset_list, dtype=np.float64)
        )

        # 抓取放置配置參數
        self.pick_place_config = PickPlaceConfig(
            pre_grasp_offset=self.declare_parameter('pre_grasp_offset', 0.1, read_only).value,
            pre_place_offset=self.declare_parameter('pre_place_offset', 0.3, read_only).value,
            arrival_tolerance=self.declare_parameter('arrival_tolerance', 0.001, read_only).value,
            stabilize_time=self.declare_parameter('stabilize_time', 2.0, read_only).value
        )

        # 其他參數
        self.pose_take_photo_static = self.declare_parameter(
            'pose_take_photo_static', [0.36, -0.44, 0.61, -pi, 0.0, pi/4]).value

        # 手動控制參數
        self.manual_config = ManualControlConfig(
            speed=self.declare_parameter('manual_speed', 5.0).value,
            speed_min=self.declare_parameter('manual_speed_min', 1.0).value,
            speed_max=self.declare_parameter('manual_speed_max', 20.0).value,
            speed_step=self.declare_parameter('manual_speed_step', 1.0).value,
            enabled=self.declare_parameter('manual_mode', False).value
        )

        # 文件路徑配置
        self.file_paths = FilePaths()

        # 服務超時配置
        self.service_timeouts = ServiceTimeouts(
            ai_inference_timeout=self.declare_parameter('ai_inference_timeout', 5.0).value,
            report_success_timeout=self.declare_parameter('report_success_timeout', 5.0).value,
            general_service_timeout=self.declare_parameter('general_service_timeout', 5.0).value
        )

    def _setup_data_storage(self):
        """設置數據存儲"""
        self.rgb_image = None
        self.depth_image = None
        self.pointcloud_data = None
        self.tm12_feedback = None

    def _setup_components(self):
        """設置各個組件"""
        # 服務管理器
        self.service_manager = ServiceManager(self, self.mutually_exclusive_cb_group)

        # 機器人控制器
        self.robot_controller = RobotController(
            self, self.service_manager, self.camera_config, self.hand_eye_config, self.robot_config
        )

        # 夾爪控制器
        self.gripper_controller = GripperController(self, self.service_manager)

        # 手動控制器
        self.manual_controller = ManualController(
            self, self.robot_controller, self.gripper_controller, self.manual_config
        )

        # 動作映射
        self.action_map = {
            "Calibration": self._execute_calibration,
            "AI_Action": self._execute_ai_action, 
            "Place_at": self._execute_ai_action_v2
        }

    def _setup_ros_interfaces(self):
        """設置 ROS 接口"""
        self._setup_subscribers()
        self._setup_publishers()
        self._setup_services()
        self._setup_action_servers()

    def _setup_subscribers(self):
        """設置訂閱者"""
        # 相機訂閱
        self.create_subscription(
            Image, "arm/camera/realsense_camera/color/image_raw",
            self._rgb_callback, 10, callback_group=self.reentrant_cb_group
        )

        self.create_subscription(
            Image, "arm/camera/realsense_camera/aligned_depth_to_color/image_raw",
            self._depth_callback, 10, callback_group=self.reentrant_cb_group
        )

        self.create_subscription(
            PointCloud2, "arm/camera/realsense_camera/depth/color/points",
            self._pointcloud_callback, 10, callback_group=self.reentrant_cb_group
        )

        # 機器人反饋訂閱
        self.create_subscription(
            FeedbackState, "arm/tm12/tm_driver/feedback_states",
            self._tm12_feedback_callback, 10, callback_group=self.reentrant_cb_group
        )

        # 鍵盤輸入訂閱
        self.create_subscription(
            Char, 'keyboard_manual', self._keyboard_callback, 10
        )

    def _setup_publishers(self):
        """設置發布者"""
        # AMR 控制發布者
        self.amr_twist_publisher = self.create_publisher(
            Twist, 'amr/iamech/iamech_driver/cmd_vel', 10
        )

    def _setup_services(self):
        """設置服務"""
        # 歸零服務
        self.create_service(
            Trigger, 'arm/homing', self._homing_service_callback,
            callback_group=self.reentrant_cb_group
        )

        # 拍照服務
        self.create_service(
            Trigger, 'arm/take_picture', self._take_picture_service_callback,
            callback_group=self.reentrant_cb_group
        )

        # 手動模式服務
        self.create_service(
            Trigger, 'arm/enable_manual_mode', self._enable_manual_mode_callback,
            callback_group=self.reentrant_cb_group
        )

        self.create_service(
            Trigger, 'arm/disable_manual_mode', self._disable_manual_mode_callback,
            callback_group=self.reentrant_cb_group
        )

    def _setup_action_servers(self):
        """設置動作服務器"""
        self.ai_action_server = ActionServer(
            self, Dotask, 'arm/ai_action', self._execute_action_callback,
            callback_group=self.reentrant_cb_group,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback
        )

    # 回調函數
    def _rgb_callback(self, msg: Image):
        """RGB 圖像回調"""
        try:
            self.rgb_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'RGB 圖像回調錯誤: {e}')

    def _depth_callback(self, msg: Image):
        """深度圖像回調"""
        try:
            self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg, "32FC1")
            self.depth_image /= 1000.0
        except Exception as e:
            self.get_logger().error(f'深度圖像回調錯誤: {e}')

    def _pointcloud_callback(self, msg: PointCloud2):
        """點雲回調"""
        try:
            self.pointcloud_data = rnp.numpify(msg)
        except Exception as e:
            self.get_logger().error(f'點雲回調錯誤: {e}')

    def _tm12_feedback_callback(self, msg: FeedbackState):
        """TM12 反饋回調"""
        try:
            self.tm12_feedback = msg
            self.robot_controller.set_feedback(msg)
        except Exception as e:
            self.get_logger().error(f'TM12 反饋回調錯誤: {e}')

    def _keyboard_callback(self, msg: Char):
        """鍵盤輸入回調"""
        try:
            key_char = chr(msg.data) if msg.data < 128 else str(msg.data)
            self.manual_controller.process_key(key_char)
        except Exception as e:
            self.get_logger().error(f'鍵盤回調錯誤: {e}')

    def _timer_callback(self):
        """定時器回調"""
        # 可以在這裡添加定期檢查的邏輯
        if self.tm12_feedback.robot_error:
            self.get_logger().warn('檢測到手臂錯誤，嘗試停止當前動作，請檢查機器人狀態')
            if hasattr(self, 'ai_action_server') and self.ai_action_server is not None:
                self.get_logger().warn('主動取消 action')

    def _parameter_callback(self, parameters) -> SetParametersResult:
        """參數更新回調"""
        result = SetParametersResult(successful=True)

        try:
            for parameter in parameters:
                if parameter.name == 'manual_mode':
                    self.manual_config.enabled = parameter.value
                    self.get_logger().info(
                        f'手動模式已{"啟用" if self.manual_config.enabled else "停用"}'
                    )
                    if self.manual_config.enabled:
                        self.get_logger().info('請開啟鍵盤節點以進行手動控制')
                elif parameter.name == 'manual_speed':
                    old_speed = self.manual_config.speed
                    self.manual_config.speed = max(
                        self.manual_config.speed_min,
                        min(parameter.value, self.manual_config.speed_max)
                    )
                    self.get_logger().info(
                        f'手動速度從 {old_speed:.1f} 更新為 {self.manual_config.speed:.1f}'
                    )

                elif parameter.name == 'pose_take_photo_static':
                    if len(parameter.value) != 6:
                        raise ValueError("拍照位置必須包含6個元素")
                    self.pose_take_photo_static = parameter.value
                    self.get_logger().info(f'拍照位置已更新: {parameter.value}')

        except Exception as e:
            self.get_logger().error(f'參數更新失敗: {e}')
            result.successful = False
            result.reason = str(e)

        return result

    # 服務回調
    def _homing_service_callback(self, request, response):
        """歸零服務回調"""
        try:
            self.robot_controller.move_to_home()
            self.gripper_controller.open_gripper()
            response.success = True
            response.message = "歸零動作完成"
        except Exception as e:
            self.get_logger().error(f'歸零失敗: {e}')
            response.success = False
            response.message = f"歸零失敗: {str(e)}"
        return response

    def _take_picture_service_callback(self, request, response):
        """拍照服務回調"""
        try:
            self._take_picture()
            response.success = True
            response.message = "拍照完成"
        except Exception as e:
            self.get_logger().error(f'拍照失敗: {e}')
            response.success = False
            response.message = f"拍照失敗: {str(e)}"
        return response

    def _enable_manual_mode_callback(self, request, response):
        """啟用手動模式回調"""
        try:
            self.manual_config.enabled = True
            response.success = True
            response.message = "手動模式已啟用"
        except Exception as e:
            self.get_logger().error(f'啟用手動模式失敗: {e}')
            response.success = False
            response.message = f"啟用手動模式失敗: {str(e)}"
        return response

    def _disable_manual_mode_callback(self, request, response):
        """停用手動模式回調"""
        try:
            self.manual_config.enabled = False
            self._stop_amr()
            response.success = True
            response.message = "手動模式已停用"
        except Exception as e:
            self.get_logger().error(f'停用手動模式失敗: {e}')
            response.success = False
            response.message = f"停用手動模式失敗: {str(e)}"
        return response

    # 動作服務器回調
    def _goal_callback(self, goal_request):
        """目標回調"""
        if goal_request.task in ['Calibration', 'Verify_Calibration', 'AI_Action']:
            return GoalResponse.ACCEPT
        return GoalResponse.REJECT

    def _cancel_callback(self, goal_handle):
        """取消回調"""
        self.get_logger().info('收到取消請求')
        if self.tm12_feedback.robot_error:
            self.get_logger().warn('檢測到手臂錯誤')
        self.robot_controller.move_to_home()
        return CancelResponse.ACCEPT

    def _execute_action_callback(self, goal_handle: ServerGoalHandle):
        """執行動作回調"""
        if self.manual_config.enabled:
            self.get_logger().warn('手動模式下無法執行自動動作')
            goal_handle.abort()
            return Dotask.Result()
        if self.tm12_feedback and self.tm12_feedback.robot_error:
            self.get_logger().warn('檢測到手臂錯誤，動作中止')
            goal_handle.abort()
            result = Dotask.Result()
            result.ok = False
            result.result = 'Robot error triggered'
            return result
        result = Dotask.Result()
        try:
            task = goal_handle.request.task
            scenario = goal_handle.request.scenario
            repeat_times = goal_handle.request.repeat_times
            if task in self.action_map:
                self.action_map[task](scenario, repeat_times)
                result.ok = True
                result.result = 'Success'
            else:
                goal_handle.abort()
                result.ok = False
                result.result = 'Unknown task'
                return result
            goal_handle.succeed()
            return result
        except Exception as e:
            self.get_logger().error(f'執行動作失敗: {e}')
            goal_handle.abort()
            result.ok = False
            result.result = str(e)
            return result

    def _execute_calibration(self, scenario: str, repeat_times: int = 1):
        """執行校正"""
        try:
            self.get_logger().info('開始執行相機校正任務...')

            calibrator = TM12Calibration(self.file_paths.calibration_config_path)

            trajectory = calibrator.get_trajectory()
            self.get_logger().info(f'校正軌跡包含 {len(trajectory)} 個位置')

            for i, pose in enumerate(trajectory):
                if self.tm12_feedback and self.tm12_feedback.robot_error:
                    self.get_logger().warn('檢測到手臂錯誤，校正中止')
                    raise Exception('Robot error triggered')
                self.get_logger().info(f'移動到校正位置 {i+1}/{len(trajectory)}')

                self.robot_controller.move_to_pose(pose, velocity=1.0)
                self.robot_controller.wait_for_arrival(pose, tolerance=0.01)

                time.sleep(self.pick_place_config.stabilize_time)  # 等待穩定

                if self.tm12_feedback is None:
                    raise RuntimeError("TM12 反饋不可用")

                actual_pose = list(self.tm12_feedback.tool_pose)

                if self.rgb_image is None or self.depth_image is None:
                    raise RuntimeError("相機影像不可用")

                success = calibrator.append_data(
                    self.rgb_image.copy(), self.depth_image.copy(), actual_pose
                )

                if not success:
                    self.get_logger().error(f'無法添加校正數據，位置 {i+1}')
                    continue

                self.get_logger().info(f'成功收集校正數據，位置 {i+1}')

            self.robot_controller.move_to_home()

            success = calibrator.execute()
            if success:
                result_path = calibrator.get_result_path()
                self.get_logger().info(f'校正成功完成！結果保存在: {result_path}')
            else:
                raise RuntimeError("校正計算失敗")

        except Exception as e:
            self.get_logger().error(f'校正失敗: {e}')
            self.robot_controller.move_to_home()
            raise

    def _execute_ai_action(self, scenario: str, repeat_times: int = 1):
        """執行 AI 動作"""
        try:
            self.get_logger().info(f'開始執行 AI 動作: "{scenario}"，重複 {repeat_times} 次')
            counter = 0
            while counter < repeat_times:
                if self.tm12_feedback and self.tm12_feedback.robot_error:
                    self.get_logger().warn('檢測到手臂錯誤，AI動作中止')
                    raise Exception('Robot error triggered')
                task_id = uuid.uuid4().hex
                self.get_logger().info(f'執行 AI 動作第 {counter+1}/{repeat_times} 次，任務 ID: {task_id}')

                # 移動到拍照位置
                self.robot_controller.move_to_pose(self.pose_take_photo_static)
                self.get_logger().info(f'debug1')
                self.robot_controller.wait_for_arrival(self.pose_take_photo_static)
                time.sleep(1)

                # 獲取機器人當前位置
                gripper_pose = self.tm12_feedback.tool_pose
                self.get_logger().info(f'debug2')

                # 調用 AI 推理
                ai_response = self._call_ai_inference(task_id)
                self.get_logger().info(f'debug3')

                if ai_response is None or not ai_response.success:
                    self.get_logger().error(f'AI 推理失敗，任務 ID: {task_id}')
                    if task_id:
                        self._report_success_to_ai(task_id, 0)
                    continue
                
                # 計算物體在基座中的位置
                object_pose_in_base = RobotPose.get_object_pose_in_base(
                    gripper_pose, ai_response.result_pose,
                    self.camera_config, self.hand_eye_config
                )

                # 執行抓取
                success, status_code = self._pick_at_pose(object_pose_in_base, 1.0)
                self.get_logger().info(f'debug4')
                if success:
                    self.get_logger().info(f'抓取成功，狀態碼: {status_code}')
                    self._report_success_to_ai(task_id, status_code)
                    # 是奎的位置
                    # Aruco


                    # 放置物體
                    self._place_at_pose(self.pick_place_config.default_place_pose)
                else:
                    self.get_logger().error(f'抓取失敗，狀態碼: {status_code}')
                    self._report_success_to_ai(task_id, status_code)
                self.get_logger().info(f'debug5')
                counter += 1
                time.sleep(0.5)
            # if repeat_times == -1:
            #     workspace_c23322231._execute_ai_action_auto_decide(self, scenario)
            self.robot_controller.move_to_home()

        except Exception as e:
            self.get_logger().error(f'AI動作失敗: {e}')
            self.robot_controller.move_to_home()
            raise

    def _call_ai_inference(self, task_id: str, text_input: str = ""):
        """調用 AI 推理服務"""
        try:
            if not self.service_manager.ai_inference.wait_for_service(
                timeout_sec=self.service_timeouts.ai_inference_timeout):
                self.get_logger().error('AI 推理服務不可用')
                return None

            request = Infer.Request()
            request.task_id = task_id
            request.text_input = text_input

            response = self.service_manager.ai_inference.call(request)
            return response

        except Exception as e:
            self.get_logger().error(f'AI 推理服務調用失敗: {e}')
            return None

    def _report_success_to_ai(self, task_id: str, status_code: int = 3):
        """向 AI 報告成功"""
        try:
            if not self.service_manager.report_success.wait_for_service(
                timeout_sec=self.service_timeouts.report_success_timeout):
                self.get_logger().error('報告成功服務不可用')
                return None

            request = Success.Request()
            request.task_id = task_id
            request.gripper_status_code = status_code

            return self.service_manager.report_success.call(request)

        except Exception as e:
            self.get_logger().error(f'報告成功服務調用失敗: {e}')
            return None

    def _pick_at_pose(self, object_pose: List[float], openning: float) -> tuple:
        """在指定位置抓取物體"""
        try:
            # 計算預抓取位置
            pre_grasp_pose = self._calculate_pre_approach_pose(
                object_pose, self.pick_place_config.pre_grasp_offset
            )

            # 打開夾爪
            self.gripper_controller.open_gripper()

            # 移動到預抓取位置
            self.robot_controller.move_to_pose(pre_grasp_pose)

            # 移動到目標位置
            self.robot_controller.move_to_pose(object_pose, MotionType.LINE_T)
            self.robot_controller.wait_for_arrival(object_pose, self.pick_place_config.arrival_tolerance)

            # 關閉夾爪
            self.gripper_controller.set_gripper_state(
                position=int(openning * 255 + 0.5), wait_time=0
            )

            # 回到預抓取位置
            self.robot_controller.move_to_pose(pre_grasp_pose, MotionType.LINE_T)

            # 獲取夾爪狀態
            status = self.gripper_controller.get_gripper_status()

            return status.ok, status.status_code

        except Exception as e:
            self.get_logger().error(f'抓取失敗: {e}')
            self.robot_controller.move_to_home()
            raise

    def _place_at_pose(self, place_pose: List[float]):
        """在指定位置放置物體"""
        try:
            # 計算預放置位置
            pre_place_pose = self._calculate_pre_approach_pose(
                place_pose, self.pick_place_config.pre_place_offset
            )

            # 移動到預放置位置
            self.robot_controller.move_to_pose(pre_place_pose)

            # 移動到放置位置
            self.robot_controller.move_to_pose(place_pose, MotionType.LINE_T)
            self.robot_controller.wait_for_arrival(place_pose, self.pick_place_config.arrival_tolerance)

            # 打開夾爪
            self.gripper_controller.open_gripper()

            # 回到預放置位置
            self.robot_controller.move_to_pose(pre_place_pose, MotionType.LINE_T)

            return True

        except Exception as e:
            self.get_logger().error(f'放置失敗: {e}')
            self.robot_controller.move_to_home()
            raise

    def _take_picture(self):
        """拍照功能"""
        try:
            if self.rgb_image is None or self.depth_image is None:
                self.get_logger().warn('RGB 或深度影像尚未準備好')
                return

            rgb = self.rgb_image.copy()
            depth = self.depth_image.copy()

            # 建立目錄
            rgb_dir = self.file_paths.rgb_dir
            depth_dir = self.file_paths.depth_dir
            os.makedirs(rgb_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)

            # 計算檔案索引
            rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith('.png')]
            depth_files = [f for f in os.listdir(depth_dir) if f.endswith('.npz')]

            rgb_index = len(rgb_files) + 1
            depth_index = len(depth_files) + 1

            # 儲存檔案
            rgb_filename = os.path.join(rgb_dir, f'rgb_{rgb_index:04d}.png')
            depth_npz_filename = os.path.join(depth_dir, f'depth_{depth_index:04d}.npz')

            cv2.imwrite(rgb_filename, rgb)
            np.savez(depth_npz_filename, depth=depth)

            self.get_logger().info(f'已儲存 RGB 影像: {rgb_filename}')
            self.get_logger().info(f'已儲存深度影像: {depth_npz_filename}')

        except Exception as e:
            self.get_logger().error(f'拍照失敗: {e}')
            raise

    def _stop_amr(self):
        """停止 AMR 移動"""
        try:
            twist_msg = Twist()
            self.amr_twist_publisher.publish(twist_msg)
            self.get_logger().info('AMR 已停止')
        except Exception as e:
            self.get_logger().error(f'停止 AMR 失敗: {e}')

    # 工具方法
    def _calculate_pre_approach_pose(self, target_pose: List[float], offset: float) -> List[float]:
        """計算預接近位置 - 用於抓取和放置操作

        Args:
            target_pose: 目標位置 [x, y, z, rx, ry, rz]
            offset: 沿著Z軸負方向的偏移距離

        Returns:
            預接近位置 [x, y, z, rx, ry, rz]
        """
        T_target = RobotPose.from_list(target_pose).to_matrix()
        approach_vector = T_target[:3, 2]  # Z軸方向向量
        T_pre = T_target.copy()
        T_pre[:3, 3] -= approach_vector * offset  # 沿Z軸負方向偏移
        return RobotPose.from_matrix(T_pre).to_list()

    def _camera_info_callback(self, msg: CameraInfo):
        """CameraInfo topic callback，僅執行一次"""
        matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        distortion = np.array(msg.d, dtype=np.float64)
        self.camera_config = CameraConfig(matrix=matrix, distortion=distortion)
        self.get_logger().info(f"CameraInfo 已獲取: matrix={matrix}, distortion={distortion}")
        # 只需獲取一次，之後可自動取消訂閱
        if hasattr(self, '_camera_info_sub') and self._camera_info_sub:
            self.destroy_subscription(self._camera_info_sub)
            self._camera_info_sub = None


def main(args=None):
    """主函數"""
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()

    node = TM12AMMNode()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        raise e
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
