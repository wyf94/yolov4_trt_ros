import os
import launch
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import EnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    param_dir = os.path.join(get_package_share_directory('yolov4_trt'), 'config', 'params.cpp.yaml')
    print(param_dir)

    return LaunchDescription([
        Node(
            package='yolov4_trt',
            executable='yolov4_trt_node',
            name='yolov4_trt_node',
            prefix=['stdbuf -o L'],
            output='screen',
            parameters=[
                {
                    param_dir
                }
            ]
         )
    ])