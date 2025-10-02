#!/usr/bin/env python3

import os
import cv2
import numpy as np
import yaml
import time
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import logging

class CalibrationResult:
    """校正結果數據結構"""
    def __init__(self):
        self.error = 0.0
        self.camera_matrix = np.eye(3)
        self.dist_coeffs = np.zeros(5)
        self.hand_eye_rotation = np.eye(3)
        self.hand_eye_translation = np.zeros(3)

class TM12Calibration:
    """TM12 機械手臂相機校正類"""

    def __init__(self, config_path: str):
        """
        初始化校正器

        Args:
            config_path: 配置文件路徑
        """
        self.config_path = Path(config_path)
        self.result_path = ""
        self.image_dir = ""
        self.timestamp = ""

        # 影像數據
        self.rgb_images: List[np.ndarray] = []
        self.depth_images: List[np.ndarray] = []
        self.robot_poses: List[List[float]] = []

        # 校正相關數據
        self.object_points: List[List[np.ndarray]] = []
        self.object_corners: List[np.ndarray] = []
        self.image_points: List[List[np.ndarray]] = []
        self.image_points_depth: List[List[float]] = []
        self.rvecs_camera: List[np.ndarray] = []
        self.tvecs_camera: List[np.ndarray] = []

        # 棋盤格參數
        self.pattern_width = 0
        self.pattern_height = 0
        self.square_size = 0.0
        self.trajectories: List[List[float]] = []

        # 校正結果
        self.result = CalibrationResult()
        self.base_points: List[List[np.ndarray]] = []

        # 設置日誌
        self.logger = logging.getLogger(__name__)

        # 讀取配置
        if not self._read_config():
            raise RuntimeError("Failed to read calibration config")

        self._initialize_object_corners()

    def _read_config(self) -> bool:
        """讀取配置文件"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            if 'tm12_amm' not in config:
                raise RuntimeError("Cannot find tm12_amm node in config file")

            tm12_config = config['tm12_amm']

            if 'sampleTrajectory' not in tm12_config or 'chessboard_info' not in tm12_config:
                raise RuntimeError("Missing required configuration fields")
            # 讀取採樣軌跡
            self.trajectories = tm12_config['sampleTrajectory']

            # 讀取棋盤格資訊
            chessboard = tm12_config['chessboard_info']
            self.pattern_width = chessboard['pattern_width']
            self.pattern_height = chessboard['pattern_height']
            self.square_size = chessboard['square_size']

            return True

        except Exception as e:
            self.logger.error(f"Failed to read config: {e}")
            return False

    def _initialize_object_corners(self):
        """初始化物體角點"""
        self.object_corners = []
        for i in range(self.pattern_height):
            for j in range(self.pattern_width):
                self.object_corners.append(np.array([
                    j * self.square_size,
                    i * self.square_size,
                    0.0
                ], dtype=np.float32))

    def _set_timestamp(self):
        """設置時間戳"""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_rotation_from_tm12(self, rx: float, ry: float, rz: float) -> np.ndarray:
        """
        從 TM12 歐拉角獲取旋轉矩陣

        Args:
            rx, ry, rz: 歐拉角（弧度）

        Returns:
            3x3 旋轉矩陣
        """
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

        return Rz @ Ry @ Rx

    def _validate_data(self):
        """驗證數據"""
        if not self.rgb_images or not self.depth_images or not self.robot_poses:
            raise RuntimeError("No images or poses available for processing")

        if len(self.rgb_images) != len(self.depth_images) or \
           len(self.rgb_images) != len(self.robot_poses):
            raise RuntimeError("Inconsistent data sizes")

    def _create_directories(self) -> bool:
        """創建目錄結構"""
        try:
            save_dir = Path("/home/robotics/hardware_tm12_amm/src/tm12_amm/tm12_amm/data/calibration")
            save_dir = save_dir / self.timestamp
            image_dir = save_dir / "images"

            save_dir.mkdir(parents=True, exist_ok=True)
            image_dir.mkdir(parents=True, exist_ok=True)

            self.result_path = str(save_dir)
            self.image_dir = str(image_dir)

            return True

        except Exception as e:
            self.logger.error(f"Failed to create directories: {e}")
            return False

    def _process_images(self) -> bool:
        """處理影像並尋找棋盤格角點"""
        try:
            pattern_size = (self.pattern_width, self.pattern_height)
            valid_image_count = 0

            for i, rgb_img in enumerate(self.rgb_images):
                # 轉換為灰階
                gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

                # 尋找棋盤格角點
                ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

                if ret:
                    # 收集角點深度資訊
                    corner_depths = []
                    for corner in corners:
                        x, y = int(corner[0][0]), int(corner[0][1])
                        depth = self.depth_images[i][y, x]
                        corner_depths.append(depth)

                    # 精細化角點位置
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                    # 儲存數據
                    self.image_points.append(corners)
                    self.image_points_depth.append(corner_depths)
                    self.object_points.append(np.array(self.object_corners))

                    valid_image_count += 1

                    # 儲存影像
                    image_filename = f"calibration_{valid_image_count}"
                    cv2.imwrite(f"{self.image_dir}/{image_filename}_rgb.jpg", rgb_img)
                    cv2.imwrite(f"{self.image_dir}/{image_filename}_gray.jpg", gray)

                    # 繪製角點
                    corner_img = rgb_img.copy()
                    cv2.drawChessboardCorners(corner_img, pattern_size, corners, ret)
                    cv2.imwrite(f"{self.image_dir}/{image_filename}_corners.jpg", corner_img)

                    # 儲存位姿資訊
                    pose_data = {
                        f"image_{valid_image_count}": {
                            "robot_pose": self.robot_poses[i],
                            "num_corners": len(corners),
                            "corners(u,v), z": [[corner[0][0], corner[0][1], depth]
                                              for corner, depth in zip(corners, corner_depths)]
                        }
                    }

                    with open(f"{self.image_dir}/{image_filename}_pose.yaml", 'w') as f:
                        yaml.dump(pose_data, f)

            return len(self.image_points) > 0

        except Exception as e:
            self.logger.error(f"Failed to process images: {e}")
            return False

    def _calculate_camera_matrix(self) -> bool:
        """計算相機矩陣和畸變係數"""
        try:
            image_size = self.rgb_images[0].shape[:2][::-1]  # (width, height)

            # 相機校正
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                self.object_points,
                self.image_points,
                image_size,
                None,
                None,
                flags=cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
            )

            self.result.error = ret
            self.result.camera_matrix = camera_matrix
            self.result.dist_coeffs = dist_coeffs
            self.rvecs_camera = rvecs
            self.tvecs_camera = tvecs

            self.logger.info(f"Camera calibration error: {ret}")
            self.logger.info(f"Camera matrix:\n{camera_matrix}")
            self.logger.info(f"Distortion coefficients: {dist_coeffs}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to calculate camera matrix: {e}")
            return False

    def _perform_hand_eye_calibration(self) -> bool:
        """執行手眼校正"""
        try:
            # 檢查數據
            if not self.rvecs_camera or not self.tvecs_camera or not self.robot_poses:
                raise RuntimeError("Empty calibration data")

            n = min(len(self.rvecs_camera), len(self.tvecs_camera), len(self.robot_poses))

            # 準備相機姿態數據
            camera_rotations = []
            camera_translations = []

            for i in range(n):
                R, _ = cv2.Rodrigues(self.rvecs_camera[i])
                camera_rotations.append(R)
                camera_translations.append(self.tvecs_camera[i])

            # 準備機器人姿態數據
            robot_rotations = []
            robot_translations = []

            for i in range(n):
                pose = self.robot_poses[i]
                R = self._get_rotation_from_tm12(pose[3], pose[4], pose[5])
                t = np.array([[pose[0]], [pose[1]], [pose[2]]])

                robot_rotations.append(R)
                robot_translations.append(t)

            # 執行手眼校正
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                robot_rotations,
                robot_translations,
                camera_rotations,
                camera_translations,
                method=cv2.CALIB_HAND_EYE_TSAI
            )

            self.result.hand_eye_rotation = R_cam2gripper
            self.result.hand_eye_translation = t_cam2gripper.flatten()

            return True

        except Exception as e:
            self.logger.error(f"Failed to perform hand-eye calibration: {e}")
            return False

    def _calculate_chessboard_corners(self) -> bool:
        """計算棋盤格角點在基座標系中的位置"""
        try:
            for frame in range(len(self.image_points)):
                camera_points = []

                # 反投影到相機座標系
                for i, corner in enumerate(self.image_points[frame]):
                    depth = self.image_points_depth[frame][i]
                    if depth <= 0:
                        continue

                    u, v = corner[0]
                    x = (u - self.result.camera_matrix[0, 2]) / self.result.camera_matrix[0, 0]
                    y = (v - self.result.camera_matrix[1, 2]) / self.result.camera_matrix[1, 1]

                    camera_points.append(np.array([x * depth, y * depth, depth]))

                # 轉換到基座標系
                T_cam2gripper = np.eye(4)
                T_cam2gripper[:3, :3] = self.result.hand_eye_rotation
                T_cam2gripper[:3, 3] = self.result.hand_eye_translation

                pose = self.robot_poses[frame]
                T_gripper2base = np.eye(4)
                T_gripper2base[:3, :3] = self._get_rotation_from_tm12(pose[3], pose[4], pose[5])
                T_gripper2base[:3, 3] = pose[:3]

                T_total = T_gripper2base @ T_cam2gripper

                base_points_frame = []
                for pt in camera_points:
                    pt_cam = np.array([pt[0], pt[1], pt[2], 1])
                    pt_base = T_total @ pt_cam
                    base_points_frame.append(pt_base[:3] / pt_base[3])

                self.base_points.append(base_points_frame)

            return True

        except Exception as e:
            self.logger.error(f"Failed to calculate chessboard corners: {e}")
            return False

    def _save_results(self) -> bool:
        """儲存校正結果"""
        try:
            # 準備校正結果數據
            calib_result = {
                'error': float(self.result.error),
                'camera_matrix': self.result.camera_matrix.tolist(),
                'distortion_coefficients': self.result.dist_coeffs.tolist(),
                'hand_eye_rotation': self.result.hand_eye_rotation.tolist(),
                'hand_eye_translation': self.result.hand_eye_translation.tolist()
            }

            # 儲存校正結果
            with open(f"{self.result_path}/calibration_result.yaml", 'w') as f:
                yaml.dump(calib_result, f)

            # 儲存角點位置數據
            corners_data = {'corners_in_base_frame': {}}
            for frame, corners in enumerate(self.base_points):
                corners_data['corners_in_base_frame'][f'frame_{frame + 1}'] = \
                    [corner.tolist() for corner in corners]

            with open(f"{self.result_path}/corners_in_base_frame.yaml", 'w') as f:
                yaml.dump(corners_data, f)

            self.logger.info(f"Calibration results saved to {self.result_path}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            return False

    def append_data(self, rgb_img: np.ndarray, depth_img: np.ndarray, robot_pose: List[float]) -> bool:
        """
        添加校正數據

        Args:
            rgb_img: RGB 影像
            depth_img: 深度影像
            robot_pose: 機器人位姿 [x, y, z, rx, ry, rz]

        Returns:
            是否成功添加
        """
        try:
            self.rgb_images.append(rgb_img.copy())
            self.depth_images.append(depth_img.copy())
            self.robot_poses.append(robot_pose.copy())
            return True
        except Exception as e:
            self.logger.error(f"Failed to append data: {e}")
            return False

    def get_trajectory(self) -> List[List[float]]:
        """
        獲取校正軌跡

        Returns:
            軌跡點列表，每個點為 [x, y, z, rx, ry, rz]
        """
        return self.trajectories.copy()

    def get_result_path(self) -> str:
        """獲取結果路徑"""
        return self.result_path

    def execute(self) -> bool:
        """
        執行校正流程

        Returns:
            是否成功完成校正
        """
        try:
            self._validate_data()
            self._set_timestamp()

            if not (self._create_directories() and
                    self._process_images() and
                    self._calculate_camera_matrix() and
                    self._perform_hand_eye_calibration() and
                    self._calculate_chessboard_corners() and
                    self._save_results()):
                return False

            return True

        except Exception as e:
            self.logger.error(f"Exception in execute: {e}")
            return False
