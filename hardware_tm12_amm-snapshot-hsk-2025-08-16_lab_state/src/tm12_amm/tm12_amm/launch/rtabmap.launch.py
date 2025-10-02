#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

#要遙控時, 請輸入
#   ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=robot/amr/iamech/iamech_driver/cmd_vel


def generate_launch_description():
    # 定義參數
    cfg = LaunchConfiguration('cfg', default='/workspaces/AI_Robot_ws/Hardware/src/ncku_csie_rl/tm12_amm/config/rtabmap_config.yaml')
    odom_frame_id = LaunchConfiguration('odom_frame_id', default='odom')
    approx_sync = LaunchConfiguration('approx_sync', default='true')
    rgb_topic = LaunchConfiguration('rgb_topic', default='/robot/amr/camera/amr_realsense/color/image_raw')
    depth_topic = LaunchConfiguration('depth_topic', default='/robot/amr/camera/amr_realsense/aligned_depth_to_color/image_raw')
    camera_info_topic = LaunchConfiguration('camera_info_topic', default='/robot/amr/camera/amr_realsense/color/camera_info')
    subscribe_scan = LaunchConfiguration('subscribe_scan', default='true')
    scan_topic = LaunchConfiguration('scan_topic', default='/robot/amr/velodyne/scan_filtered')
    visual_odometry = LaunchConfiguration('visual_odometry', default='false')
    odom_topic = LaunchConfiguration('odom_topic', default='/robot/amr/iamech/iamech_odom/odom')
    rtabmap_args = LaunchConfiguration('rtabmap_args', default='-d')
    rtabmap_viz = LaunchConfiguration('rtabmap_viz', default='true')
    rviz = LaunchConfiguration('rviz', default='true')

    # 包含 rtabmap.launch.py 檔案
    rtabmap_launch_dir = os.path.join(get_package_share_directory('rtabmap_launch'), 'launch')
    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([rtabmap_launch_dir, '/rtabmap.launch.py']),
        launch_arguments={
            'cfg': cfg,
            'odom_frame_id': odom_frame_id,
            'approx_sync': approx_sync,
            'rgb_topic': rgb_topic,
            'depth_topic': depth_topic,
            'camera_info_topic': camera_info_topic,
            'subscribe_scan': subscribe_scan,
            'scan_topic': scan_topic,
            'visual_odometry': visual_odometry,
            'odom_topic': odom_topic,
            'rtabmap_args': rtabmap_args,
            'rtabmap_viz': rtabmap_viz,
            'rviz': rviz,
            'Rtabmap/DetectionRate': '10'
            # 以下是原始 XML 中被註解的選項，如需啟用請取消註解
            # 'publish_tf_odom': 'false',  # 避免與 robot_localization 的 odom tf 衝突
            # 'subscribe_odom_info': 'false',  # 使用外部里程計
            # 'subscribe_depth': 'false',  # 嘗試修復訊息丟失問題
        }.items()
    )

    # 返回啟動描述
    return LaunchDescription([
        rtabmap_launch,
    ])