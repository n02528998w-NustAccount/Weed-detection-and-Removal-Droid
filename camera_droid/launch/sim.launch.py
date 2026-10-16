import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('camera_droid')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    with open(os.path.join(pkg, 'urdf', 'camera_droid.urdf')) as f:
        robot_description = f.read()

    world = os.path.join(pkg, 'worlds', 'arena.sdf')
    bridge_cfg = os.path.join(pkg, 'config', 'bridge.yaml')
    weeds_file = os.path.join(pkg, 'config', 'weeds.yaml')

    args = [
        DeclareLaunchArgument('patrol', default_value='true',
                              description='Run the autonomous patrol + weed-cutting node'),
        DeclareLaunchArgument('cut_weeds', default_value='true',
                              description='Photograph and cut weeds found while patrolling'),
        DeclareLaunchArgument('x_min', default_value='-4.0'),
        DeclareLaunchArgument('x_max', default_value='4.0'),
        DeclareLaunchArgument('y_min', default_value='-3.0'),
        DeclareLaunchArgument('y_max', default_value='3.0'),
        DeclareLaunchArgument('lane_spacing', default_value='2.0'),
        DeclareLaunchArgument('photo_distance', default_value='1.5',
                              description='Front-camera photo every N metres while driving'),
        DeclareLaunchArgument('photo_dir', default_value='~/droid_photos'),
    ]

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': f'-r -v 2 {world}'}.items(),
    )

    rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        output='screen',
    )

    spawn = Node(
        package='ros_gz_sim', executable='create',
        arguments=['-name', 'camera_droid', '-topic', 'robot_description',
                   '-x', '0.0', '-y', '0.0', '-z', '0.02'],
        output='screen',
    )

    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        parameters=[{'config_file': bridge_cfg, 'use_sim_time': True}],
        output='screen',
    )

    # Lets the patrol node delete a weed model once it has been cut
    remove_bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        name='remove_entity_bridge',
        arguments=['/world/arena/remove@ros_gz_interfaces/srv/DeleteEntity'],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    detector = Node(
        package='camera_droid', executable='weed_detector',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    def f(name):
        return ParameterValue(LaunchConfiguration(name), value_type=float)

    patrol = Node(
        package='camera_droid', executable='area_patrol',
        condition=IfCondition(LaunchConfiguration('patrol')),
        parameters=[{
            'use_sim_time': True,
            'x_min': f('x_min'), 'x_max': f('x_max'),
            'y_min': f('y_min'), 'y_max': f('y_max'),
            'lane_spacing': f('lane_spacing'),
            'photo_distance': f('photo_distance'),
            'photo_dir': ParameterValue(LaunchConfiguration('photo_dir'), value_type=str),
            'cut_weeds': ParameterValue(LaunchConfiguration('cut_weeds'), value_type=bool),
            'weeds_file': weeds_file,
            'world_name': 'arena',
        }],
        output='screen',
    )

    return LaunchDescription(args + [gz_sim, rsp, spawn, bridge, remove_bridge, detector, patrol])
