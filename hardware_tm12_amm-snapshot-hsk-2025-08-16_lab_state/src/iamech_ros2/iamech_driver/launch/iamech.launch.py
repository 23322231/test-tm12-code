#!/usr/bin/env python3

import sys
import os
import yaml
import launch
from launch import LaunchDescription
from launch.actions import LogInfo, IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # 添加控制器配置文件路徑
    controller_config = os.path.join(
        get_package_share_directory('iamech_driver'),
        'config',
        'iamech.yaml'
    )

    return LaunchDescription([

        Node(
            package='iamech_driver',
            executable='iamech_ros_driver.py',
            name='iamech_driver',
            output='screen',
            #arguments=[''],
        ),

        Node(
            package='rqt_gui',
            executable='rqt_gui',
            name='rqt',
            output='screen',
            #arguments=['-d', rviz_config]
        )
    ])