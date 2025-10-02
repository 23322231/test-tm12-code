#!/usr/bin/env python3

import os
import launch
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, GroupAction, DeclareLaunchArgument  # 添加DeclareLaunchArgument
from launch_ros.actions import Node, PushRosNamespace
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration  # 確保導入LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    declare_log_level_arg = DeclareLaunchArgument(
        name='log-level',
        default_value='info',
        description='Logging level (info, debug, ...)'
    )

    return LaunchDescription([
        declare_log_level_arg,
        Node(
            package='iamech_driver',
            executable='iamech_ros_driver.py',
            namespace='robot/amr/iamech',
            name='iamech_driver',
            output='screen',
        ),

        Node(
            package='iamech_driver',
            executable='iamech_ros_odom.py',
            namespace='robot/amr/iamech',
            name='iamech_odom',
            output='screen',
        ),

        # RealSense 相機
        GroupAction(
            actions=[
                PushRosNamespace('robot/amr'),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource([
                        FindPackageShare('realsense2_camera'), '/launch/rs_launch.py'
                    ]),
                    launch_arguments={
                        'serial_no': "'827112072898'",
                        'align_depth.enable': 'true',
                        'pointcloud.enable': 'true',
                        'spatial_filter.enable': 'true',
                        'rgb_camera.color_profile': '1280x720x30',
                        'depth_module.depth_profile': '1280x720x30',
                        'rgb_camera.enable_auto_exposure': 'true',
                        'depth_module.enable_auto_exposure': 'true',
                        'camera_name': 'amr_realsense'
                    }.items(),
                )
            ]
        ),

        # Velodyne
        GroupAction(
            actions=[
                PushRosNamespace('robot/amr/velodyne'),  # 如果需要命名空間，可以取消註解
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource([
                        FindPackageShare('velodyne'), '/launch/velodyne-all-nodes-VLP16-launch.py'
                    ])
                )
            ]
        ),

        # Velodyne 濾波
        Node(
            package="laser_filters",
            executable="scan_to_scan_filter_chain",
            namespace='robot/amr/velodyne',
            name="vlp16_filter",
            parameters=[PathJoinSubstitution([
                FindPackageShare("tm12_amm"), "config", "vlp16_filter.yaml"
            ])],
            output='screen'  # 添加此行以查看節點的輸出信息
        ),

        # 前方 SICK S300 掃描儀 - 使用套件提供的 launch 檔案
        GroupAction(
            actions=[
                PushRosNamespace('robot/amr/sicks300_front'),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource([
                        FindPackageShare('sicks300_ros2'), '/launch/scan_with_filter.launch.py'
                    ]),
                    launch_arguments={
                        'sicks300_param_file': PathJoinSubstitution([
                            FindPackageShare("tm12_amm"), "config", "sicks300_front.yaml"
                        ]),
                        'log-level': 'info'
                    }.items(),
                )
            ]
        ),

        # 後方 SICK S300 掃描儀 - 使用套件提供的 launch 檔案
        GroupAction(
            actions=[
                PushRosNamespace('robot/amr/sicks300_back'),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource([
                        FindPackageShare('sicks300_ros2'), '/launch/scan_with_filter.launch.py'
                    ]),
                    launch_arguments={
                        'sicks300_param_file': PathJoinSubstitution([
                            FindPackageShare("tm12_amm"), "config", "sicks300_back.yaml"
                        ]),
                        'log-level': 'info'
                    }.items(),
                )
            ]
        ),

        # ira_laser_tools - 合併雷射掃描數據，修正參數格式
        Node(
            package='ira_laser_tools',
            executable='laserscan_multi_merger',
            namespace='robot/amr/laser_merger',
            name='scan_merger',
            output='screen',
            parameters=[{
                'destination_frame': 'base_link',
                'cloud_destination_topic': 'merged_cloud',
                'scan_destination_topic': 'merged_scan',
                'laserscan_topics': '/robot/amr/sicks300_back/scan /robot/amr/sicks300_front/scan',  #/robot/amr/velodyne/scan_filtered'
                'angle_min': -3.14159,
                'angle_max': 3.14159,
                'angle_increment': 0.0058,
                'scan_time': 0.033,
                'range_min': 0.1,
                'range_max': 50.0,
                'height_min': -0.8,
                'height_max': 0.2,
                'use_inf': False
            }]
        ),

        # lidar 到機器人底座
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='velodyne_to_base_tf',
            namespace='robot/tf',
            arguments=['0.4800', '-0.2772', '0.0000', '0', '0', '0', 'base_link', 'velodyne']
        ),

        # amr_rs 到機器人底座
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='amr_realsense_to_base_tf',
            namespace='robot/tf',
            arguments=['0.5300', '-0.0905', '0.0650', '0', '0', '0', 'base_link', 'amr_realsense_link']
        ),

        # 前方掃描儀到底座的 TF
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='front_s300_to_base_tf',
            namespace='robot/tf',
            arguments=['0.59623', '-0.37123', '-0.68', '-0.7854', '0.0', '3.1416', 'base_link', 's300_front']
        ),

        # 後方掃描儀到底座的 TF
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='back_s300_to_base_tf',
            namespace='robot/tf',
            arguments=['-0.59623', '0.37123', '-0.68', '2.3562', '0.0', '3.1416', 'base_link', 's300_back']
        ),

    ])