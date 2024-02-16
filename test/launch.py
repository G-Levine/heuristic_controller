
from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Get URDF via xacro
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                #    FindPackageShare("rhea_description"),
                    "/home/pi/ros2_ws/src/pupper_v3_description",
                    "description",
                    "pupper_v3.urdf.xacro",
                ]
            ),
        ]
    )
    robot_description = {"robot_description": robot_description_content}

    robot_controllers = PathJoinSubstitution(
        [
        #    FindPackageShare("neural_controller"),
            "/home/pi/ros2_ws/src/heuristic_controller",
            "test",
            "config.yaml",
        ]
    )

    gdb_prefix = 'gdb -ex run --args'

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, robot_controllers],
        output="both",
        # prefix=gdb_prefix,
    )

    robot_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["heuristic_controller", "--controller-manager", "/controller_manager", "--controller-manager-timeout", "30"],
    )

    nodes = [
        control_node,
        robot_controller_spawner,
    ]

    return LaunchDescription(nodes)
