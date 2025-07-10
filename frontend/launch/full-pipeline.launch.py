from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='visualize',
            executable='visualize',
            name='visualize',
            output='screen'
        ),

        Node(
            package='frontend',
            executable='triangulate',
            name='triangulate',
            output='screen'
        ),

        Node(
            package='frontend',
            executable='featuretrack',
            name='featuretrack',
            output='screen'
        ),
        Node(
            package='frontend',
            executable='dispmap',
            name='dispmap',
            output='screen'
        ),
        # ExecuteProcess(
        #     cmd=['ros2', 'bag', 'play', '/docker/KITTI/rosbag2'],
        #     output='screen'
        # ),

    ])
