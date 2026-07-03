#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleStatus
import time

class ServoControl(Node):
    """
    一个通过 XRCE-DDS (Micro-ROS) 直接向 PX4 发送执行器控制命令的节点。
    它会循环改变第一个执行器的输出值。
    """

    def __init__(self):
        super().__init__('actuator_control_node')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status_v1',
            self.vehicle_status_callback,
            qos_profile
        )

        self.command_pub = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            qos_profile
        )

        self.is_connected = False
        self.arming_state = 0
        self.get_logger().info('节点已初始化，正在等待飞控连接...')

    def vehicle_status_callback(self, msg):
        if not self.is_connected:
            self.is_connected = True
            self.get_logger().info('飞控已连接!')
        self.arming_state = msg.arming_state

    def publish_actuator_command(self, values):
        if len(values) > 6:
            self.get_logger().warning('输入的值超过6个，多余的值将被忽略。')
        
        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_ACTUATOR
        
        # 将列表中的值精确映射到 param1 到 param6
        msg.param1 = float(values[0]) if len(values) > 0 else 0.0
        msg.param2 = float(values[1]) if len(values) > 1 else 0.0
        msg.param3 = float(values[2]) if len(values) > 2 else 0.0
        msg.param4 = float(values[3]) if len(values) > 3 else 0.0
        msg.param5 = float(values[4]) if len(values) > 4 else 0.0
        msg.param6 = float(values[5]) if len(values) > 5 else 0.0
        # param7 在此命令中不被使用

        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 255 
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        self.command_pub.publish(msg)
        self.get_logger().info(f"发送执行器命令, param1: {msg.param1:.2f}, param2: {msg.param2:.2f}, ...")

    def open_servo(self,servo_1=0,servo_2=0):
        # 定义一个包含6个值的列表，以匹配 MAV_CMD_DO_SET_ACTUATOR 的参数
        actuator_values = [0.0] * 6
        actuator_values[0] = servo_1
        actuator_values[1] = servo_2
        self.publish_actuator_command(actuator_values)

