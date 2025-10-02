#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
import os

def generate_launch_description():
    """生成 keyboard publisher launch 描述"""

    # Launch arguments
    node_name_arg = DeclareLaunchArgument(
        'node_name',
        default_value='keyboard_publisher',
        description='鍵盤發布器節點名稱'
    )

    # Keyboard publisher node
    keyboard_publisher_node = Node(
        package='tm12_amm',
        executable='keyboard_pub.py',
        namespace='robot',
        name=LaunchConfiguration('node_name'),
        output='screen',
        parameters=[],
    )

    return LaunchDescription([
        node_name_arg,
        keyboard_publisher_node,
    ])
