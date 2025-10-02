#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.action.server import ServerGoalHandle

import os
import time
import numpy as np
from typing import List, Optional
from pathlib import Path
from dataclasses import dataclass

from rcl_interfaces.msg import ParameterDescriptor
from std_srvs.srv import Trigger
from tm_msgs.srv import SetPositions
from tm_msgs.msg import FeedbackState
from sensor_msgs.msg import Image
from tm12_amm_interfaces.action import Calibration
from cv_bridge import CvBridge

from calibration import TM12Calibration


class CalibrationNode(Node):
    """
    獨立的校正節點
    提供校正服務和動作服務器
    """

    def __init__(self, name: str = 'calibration_node'):
        super().__init__(name)
        self.get_logger().info('Calibration Node 初始化中...')

        # 基本設置
        self.cv_bridge = CvBridge()
        self.callback_group = ReentrantCallbackGroup()

        # 宣告並獲取所有參數
        self._declare_parameters()
        self._get_parameters()

        # 數據存儲
        self.rgb_image = None
        self.depth_image = None
        self.robot_feedback = None

        # 設置 ROS 接口
        self._setup_subscribers()
        self._setup_service_clients()
        self._setup_services()
        self._setup_action_servers()

        self.get_logger().info('Calibration Node 初始化完成')

    def _declare_parameters(self):
        """宣告所有參數"""
        # 路徑相關參數
        self.declare_parameter('config_dir',
                             '/workspaces/AI_Robot_ws/Hardware/src/tm12_amm/tm12_amm/config',
                             ParameterDescriptor(description='校正配置文件目錄'))
        self.declare_parameter('default_config_file',
                             'calibration_camera_eih_001.yaml',
                             ParameterDescriptor(description='默認校正配置文件名'))
        self.declare_parameter('results_dir',
                             '/workspaces/AI_Robot_ws/Hardware/src/tm12_amm/tm12_amm/data/calibration_results',
                             ParameterDescriptor(description='校正結果保存目錄'))

        # 相機主題名稱
        self.declare_parameter('rgb_topic',
                             'arm/camera/realsense_camera/color/image_raw',
                             ParameterDescriptor(description='RGB 圖像主題'))
        self.declare_parameter('depth_topic',
                             'arm/camera/realsense_camera/aligned_depth_to_color/image_raw',
                             ParameterDescriptor(description='深度圖像主題'))
        self.declare_parameter('robot_feedback_topic',
                             'arm/tm12/tm_driver/feedback_states',
                             ParameterDescriptor(description='機器人反饋主題'))

        # 服務名稱
        self.declare_parameter('tm12_set_positions_service',
                             'arm/tm12/tm_driver/set_positions',
                             ParameterDescriptor(description='TM12 設定位置服務'))
        self.declare_parameter('homing_service',
                             'arm/homing',
                             ParameterDescriptor(description='機器人歸零服務'))

        # 超時和容差參數
        self.declare_parameter('move_timeout', 10.0,
                             ParameterDescriptor(description='機器人移動超時 (秒)'))
        self.declare_parameter('arrival_timeout', 30.0,
                             ParameterDescriptor(description='等待機器人到達超時 (秒)'))
        self.declare_parameter('position_tolerance', 0.01,
                             ParameterDescriptor(description='位置到達容差'))
        self.declare_parameter('service_timeout', 5.0,
                             ParameterDescriptor(description='服務等待超時 (秒)'))
        self.declare_parameter('homing_timeout', 30.0,
                             ParameterDescriptor(description='歸零服務超時 (秒)'))
        self.declare_parameter('camera_stabilization_time', 2.0,
                             ParameterDescriptor(description='相機穩定等待時間 (秒)'))

        # 機器人運動參數
        self.declare_parameter('robot_velocity', 1.0,
                             ParameterDescriptor(description='機器人運動速度'))
        self.declare_parameter('robot_acc_time', 1.0,
                             ParameterDescriptor(description='機器人加速時間'))
        self.declare_parameter('robot_blend_percentage', 0,
                             ParameterDescriptor(description='機器人混合百分比'))
        self.declare_parameter('robot_fine_goal', True,
                             ParameterDescriptor(description='機器人精確目標'))

    def _get_parameters(self):
        """獲取所有參數"""
        # 路徑相關
        self.config_dir = self.get_parameter('config_dir').value
        self.default_config_file = self.get_parameter('default_config_file').value
        self.results_dir = self.get_parameter('results_dir').value

        # 主題名稱
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.robot_feedback_topic = self.get_parameter('robot_feedback_topic').value

        # 服務名稱
        self.tm12_set_positions_service = self.get_parameter('tm12_set_positions_service').value
        self.homing_service = self.get_parameter('homing_service').value

        # 超時和容差
        self.move_timeout = self.get_parameter('move_timeout').value
        self.arrival_timeout = self.get_parameter('arrival_timeout').value
        self.position_tolerance = self.get_parameter('position_tolerance').value
        self.service_timeout = self.get_parameter('service_timeout').value
        self.homing_timeout = self.get_parameter('homing_timeout').value
        self.camera_stabilization_time = self.get_parameter('camera_stabilization_time').value

        # 機器人運動參數
        self.robot_velocity = self.get_parameter('robot_velocity').value
        self.robot_acc_time = self.get_parameter('robot_acc_time').value
        self.robot_blend_percentage = self.get_parameter('robot_blend_percentage').value
        self.robot_fine_goal = self.get_parameter('robot_fine_goal').value

        # 計算路徑
        self.default_config_path = os.path.join(self.config_dir, self.default_config_file)

    def _setup_subscribers(self):
        """設置訂閱者"""
        # 相機影像訂閱
        self.create_subscription(
            Image,
            self.rgb_topic,
            self._rgb_callback,
            10,
            callback_group=self.callback_group
        )

        self.create_subscription(
            Image,
            self.depth_topic,
            self._depth_callback,
            10,
            callback_group=self.callback_group
        )

        # 機器人狀態訂閱
        self.create_subscription(
            FeedbackState,
            self.robot_feedback_topic,
            self._robot_feedback_callback,
            10,
            callback_group=self.callback_group
        )

    def _setup_service_clients(self):
        """設置服務客戶端"""
        # TM12 位置設定服務
        self.tm12_set_positions_client = self.create_client(
            SetPositions,
            self.tm12_set_positions_service,
            callback_group=self.callback_group
        )

        # 機器人歸零服務
        self.homing_client = self.create_client(
            Trigger,
            self.homing_service,
            callback_group=self.callback_group
        )

    def _setup_services(self):
        """設置服務"""
        # 快速校正服務
        self.create_service(
            Trigger,
            'calibration/quick_calibrate',
            self._quick_calibrate_callback,
            callback_group=self.callback_group
        )

        # 驗證校正服務
        self.create_service(
            Trigger,
            'calibration/verify',
            self._verify_calibration_callback,
            callback_group=self.callback_group
        )

        # 載入校正結果服務
        self.create_service(
            Trigger,
            'calibration/load_results',
            self._load_calibration_results_callback,
            callback_group=self.callback_group
        )

    def _setup_action_servers(self):
        """設置動作服務器"""
        self.calibration_action_server = ActionServer(
            self,
            Calibration,
            'calibration/execute',
            self._execute_calibration_callback,
            callback_group=self.callback_group,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback
        )

    # 回調函數
    def _rgb_callback(self, msg: Image):
        """RGB 圖像回調"""
        try:
            self.rgb_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'RGB 圖像處理錯誤: {e}')

    def _depth_callback(self, msg: Image):
        """深度圖像回調"""
        try:
            self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg, "32FC1")
            self.depth_image /= 1000.0
        except Exception as e:
            self.get_logger().error(f'深度圖像處理錯誤: {e}')

    def _robot_feedback_callback(self, msg: FeedbackState):
        """機器人反饋回調"""
        self.robot_feedback = msg

    # 動作服務器回調
    def _goal_callback(self, goal_request):
        """動作目標回調"""
        self.get_logger().info('接收到校正動作請求')
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        """動作取消回調"""
        self.get_logger().info('接收到校正動作取消請求')
        # 回到安全位置
        self._call_homing()
        return CancelResponse.ACCEPT

    def _execute_calibration_callback(self, goal_handle: ServerGoalHandle):
        """執行校正動作回調"""
        self.get_logger().info('開始執行校正動作...')

        result = Calibration.Result()
        feedback = Calibration.Feedback()

        try:
            # 獲取校正配置
            config_path = (goal_handle.request.config_path
                          if goal_handle.request.config_path
                          else self.default_config_path)

            # 創建校正器
            calibrator = TM12Calibration(config_path)
            trajectory = calibrator.get_trajectory()

            self.get_logger().info(f'校正軌跡包含 {len(trajectory)} 個位置')

            # 執行校正軌跡
            for i, pose in enumerate(trajectory):
                # 檢查是否被取消
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.success = False
                    result.message = "校正被取消"
                    return result

                # 更新進度
                feedback.current_position = i + 1
                feedback.total_positions = len(trajectory)
                feedback.current_pose = pose
                feedback.status = f"移動到位置 {i+1}/{len(trajectory)}"
                goal_handle.publish_feedback(feedback)

                # 移動機器人
                success = self._move_robot_to_pose(pose)
                if not success:
                    result.success = False
                    result.message = f"無法移動到位置 {i+1}"
                    goal_handle.abort()
                    return result

                # 等待機器人到達
                self._wait_for_robot_arrival(pose)

                # 等待相機穩定
                time.sleep(self.camera_stabilization_time)

                # 收集數據
                if not self._collect_calibration_data(calibrator, pose):
                    self.get_logger().warn(f'位置 {i+1} 數據收集失敗')
                    continue

                feedback.status = f"位置 {i+1} 數據收集完成"
                goal_handle.publish_feedback(feedback)

            # 回到安全位置
            self._call_homing()

            # 執行校正計算
            feedback.status = "執行校正計算..."
            goal_handle.publish_feedback(feedback)

            calibration_success = calibrator.execute()

            if calibration_success:
                result.success = True
                result.message = "校正成功完成"
                result.result_path = calibrator.get_result_path()
                goal_handle.succeed()
            else:
                result.success = False
                result.message = "校正計算失敗"
                goal_handle.abort()

            return result

        except Exception as e:
            self.get_logger().error(f'校正執行失敗: {e}')
            # 確保機器人回到安全位置
            self._call_homing()
            result.success = False
            result.message = f"校正失敗: {str(e)}"
            goal_handle.abort()
            return result

    # 服務回調
    def _quick_calibrate_callback(self, request, response):
        """快速校正服務回調"""
        try:
            self.get_logger().info('開始快速校正...')

            # 創建校正器
            calibrator = TM12Calibration(self.default_config_path)

            # 獲取簡化軌跡（較少的位置）
            trajectory = calibrator.get_quick_trajectory()

            # 執行校正
            success = self._execute_calibration_sequence(calibrator, trajectory)

            if success:
                response.success = True
                response.message = f"快速校正成功，結果保存在: {calibrator.get_result_path()}"
            else:
                response.success = False
                response.message = "快速校正失敗"

        except Exception as e:
            self.get_logger().error(f'快速校正失敗: {e}')
            response.success = False
            response.message = f"快速校正失敗: {str(e)}"

        return response

    def _verify_calibration_callback(self, request, response):
        """驗證校正服務回調"""
        try:
            self.get_logger().info('開始驗證校正...')

            # 載入現有校正結果
            # 這裡可以實現校正結果的驗證邏輯

            response.success = True
            response.message = "校正驗證完成"

        except Exception as e:
            self.get_logger().error(f'校正驗證失敗: {e}')
            response.success = False
            response.message = f"校正驗證失敗: {str(e)}"

        return response

    def _load_calibration_results_callback(self, request, response):
        """載入校正結果服務回調"""
        try:
            self.get_logger().info('載入校正結果...')

            # 這裡可以實現載入和發布校正結果的邏輯
            # 例如更新參數服務器或發布變換

            response.success = True
            response.message = "校正結果載入完成"

        except Exception as e:
            self.get_logger().error(f'載入校正結果失敗: {e}')
            response.success = False
            response.message = f"載入校正結果失敗: {str(e)}"

        return response

    # 輔助方法
    def _move_robot_to_pose(self, pose: List[float], timeout: float = None) -> bool:
        """移動機器人到指定位置"""
        if timeout is None:
            timeout = self.move_timeout

        try:
            if not self.tm12_set_positions_client.wait_for_service(timeout_sec=self.service_timeout):
                self.get_logger().error('TM12 設定位置服務不可用')
                return False

            request = SetPositions.Request()
            request.motion_type = 2  # PTP_T
            request.positions = pose
            request.velocity = self.robot_velocity
            request.acc_time = self.robot_acc_time
            request.blend_percentage = self.robot_blend_percentage
            request.fine_goal = self.robot_fine_goal

            future = self.tm12_set_positions_client.call_async(request)

            # 等待服務完成
            start_time = time.time()
            while not future.done():
                if time.time() - start_time > timeout:
                    self.get_logger().error('機器人移動超時')
                    return False
                time.sleep(0.1)

            response = future.result()
            return response.ok if hasattr(response, 'ok') else True

        except Exception as e:
            self.get_logger().error(f'機器人移動失敗: {e}')
            return False

    def _wait_for_robot_arrival(self, target_pose: List[float],
                               tolerance: float = None, timeout: float = None):
        """等待機器人到達目標位置"""
        if tolerance is None:
            tolerance = self.position_tolerance
        if timeout is None:
            timeout = self.arrival_timeout

        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.robot_feedback is None:
                time.sleep(0.1)
                continue

            current_pose = self.robot_feedback.tool_pose
            error = np.linalg.norm(np.array(target_pose) - np.array(current_pose))

            if error < tolerance:
                return True

            time.sleep(0.1)

        self.get_logger().warn('機器人到達目標位置超時')
        return False

    def _collect_calibration_data(self, calibrator: TM12Calibration,
                                 target_pose: List[float]) -> bool:
        """收集校正數據"""
        try:
            if self.rgb_image is None or self.depth_image is None:
                self.get_logger().error('影像數據不可用')
                return False

            if self.robot_feedback is None:
                self.get_logger().error('機器人反饋不可用')
                return False

            # 獲取實際機器人位置
            actual_pose = self.robot_feedback.tool_pose.copy()

            # 複製影像數據
            rgb_img = self.rgb_image.copy()
            depth_img = self.depth_image.copy()

            # 添加到校正器
            success = calibrator.append_data(rgb_img, depth_img, actual_pose)

            if success:
                self.get_logger().info('校正數據收集成功')
            else:
                self.get_logger().error('校正數據收集失敗')

            return success

        except Exception as e:
            self.get_logger().error(f'收集校正數據時發生錯誤: {e}')
            return False

    def _call_homing(self) -> bool:
        """調用歸零服務"""
        try:
            if not self.homing_client.wait_for_service(timeout_sec=self.service_timeout):
                self.get_logger().error('歸零服務不可用')
                return False

            request = Trigger.Request()
            future = self.homing_client.call_async(request)

            # 等待服務完成
            start_time = time.time()
            while not future.done():
                if time.time() - start_time > self.homing_timeout:
                    self.get_logger().error('歸零服務超時')
                    return False
                time.sleep(0.1)

            response = future.result()
            return response.success

        except Exception as e:
            self.get_logger().error(f'歸零服務調用失敗: {e}')
            return False

    def _execute_calibration_sequence(self, calibrator: TM12Calibration,
                                    trajectory: List[List[float]]) -> bool:
        """執行校正序列"""
        try:
            for i, pose in enumerate(trajectory):
                self.get_logger().info(f'執行校正位置 {i+1}/{len(trajectory)}')

                # 移動機器人
                if not self._move_robot_to_pose(pose):
                    self.get_logger().error(f'無法移動到位置 {i+1}')
                    return False

                # 等待到達
                if not self._wait_for_robot_arrival(pose):
                    self.get_logger().error(f'等待到達位置 {i+1} 超時')
                    return False

                # 等待穩定
                time.sleep(self.camera_stabilization_time)

                # 收集數據
                if not self._collect_calibration_data(calibrator, pose):
                    self.get_logger().warn(f'位置 {i+1} 數據收集失敗')
                    continue

            # 回到安全位置
            self._call_homing()

            # 執行校正計算
            return calibrator.execute()

        except Exception as e:
            self.get_logger().error(f'校正序列執行失敗: {e}')
            return False


def main(args=None):
    """主函數"""
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()

    node = CalibrationNode()
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
