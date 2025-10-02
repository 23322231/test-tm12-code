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



def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rqt_gui',
            executable='rqt_gui',
            namespace='robot/gui',
            name='rqt',
            output='screen',
            #arguments=['-d', rviz_config]
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            namespace='robot/gui',
            name='rviz2',
            output='screen',
            #arguments=['-d', rviz_config]
        )
    ])