#!/usr/bin/env python3

import os
import launch
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_dir = get_package_share_directory('tm12_amm')

    # 引用各模組的 launch 檔案
    arm_core_launch_path = os.path.join(pkg_dir, 'launch', 'robot_arm_core.launch.py')
    gui_launch_path = os.path.join(pkg_dir, 'launch', 'robot_gui.launch.py')

    camera_config = os.path.join(
        get_package_share_directory('tm12_amm'),
        'config',
        'camera_config.yaml'
    )

    return LaunchDescription([
        # 啟動手臂核心
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([arm_core_launch_path])
        ),

        Node(
            package='tm12_amm',
            executable='tm12_amm.py',
            namespace='robot',
            name='tm12_amm',
            output='screen',
            parameters=[camera_config]
        ),

        # 啟動 GUI
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([gui_launch_path])
        )
    ])