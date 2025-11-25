from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Target Controller (cf_1) - Basic MPC controller for target drone
        Node(
            package='controller_pkg',
            executable='crazyflie_mpc_controller',
            name='cf_1',
            output='screen',
            parameters=[
                # Add any parameters if needed
            ]
        ),
        
        # Pursuer Controller (cf_2) - Self-adaptive MPC controller for tracking
        Node(
            package='controller_pkg',
            executable='self_adaptive_mpc_controller',
            output='screen',
            parameters=[
                # Add any parameters if needed
            ]
        ),
        
    ])