import sys
import launch
from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='grpr2f85_driver',
            executable='grpr2f85_driver.py',
            output='screen',
            arguments=['usb_port:=0'],
        ),

        Node(
            package='rqt_gui',
            executable='rqt_gui',
            name='rqt',
            output='screen',
            #arguments=['-d', rviz_config]
        ),
    ])