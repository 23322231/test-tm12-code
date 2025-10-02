#!/usr/bin/env python3

import sys
import os
import yaml
import launch
from launch import LaunchDescription
from launch.actions import LogInfo, IncludeLaunchDescription, GroupAction
from launch_ros.actions import Node, PushRosNamespace
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    camera_config = os.path.join(
        get_package_share_directory('tm12_amm'),
        'config',
        'camera_config.yaml'
    )

    return LaunchDescription([
        Node(
            package='tm_driver',
            executable='tm_driver',
            namespace='robot/arm/tm12/tm_driver',
            output='screen',
            arguments=['robot_ip:=192.168.10.2'],
        ),

        Node(
            package='grpr2f85_driver',
            executable='grpr2f85_driver.py',
            namespace='robot/arm/gripper',
            output='screen',
            arguments=['usb_port:=0'],
        ),

        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            namespace='robot/arm/camera',
            name='realsense_camera',
            parameters=[{
                'serial_no': '925322060682',
                'align_depth.enable': True,
                'pointcloud.enable': True,
                'spatial_filter.enable': True,
                'rgb_camera.color_profile': '1280x720x30',
                'depth_module.depth_profile': '1280x720x30',
                'rgb_camera.enable_auto_exposure': True,
                'depth_module.enable_auto_exposure': True,
                'camera_name': 'arm_realsense'
            }]
        ),

        # TM12手臂基座到機器人底座
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='tm12_base_to_base_tf',
            namespace='robot/tf',
            arguments=['-0.2817', '0.0000', '0.0000', '0.7854', '0', '0', 'base_link', 'tm12_base_link']
        )
    ])