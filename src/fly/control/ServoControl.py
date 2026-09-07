#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleStatus
import time

class ServoControl(Node):
    """
    通过 XRCE-DDS 直接向 PX4 飞控发送舵机控制指令。
    MAIN5 = Actuator Set 5, MAIN6 = Actuator Set 6
    使用 MAV_CMD_DO_SET_ACTUATOR (187)。
    """

    def __init__(self):
        super().__init__('actuator_control_node')

        # 订阅使用 BEST_EFFORT — PX4 的 /fmu/out 话题用此 QoS
        sub_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status_v1',
            self.vehicle_status_callback,
            sub_qos_profile
        )

        # 发布使用 RELIABLE — 确保指令到达飞控
        pub_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.command_pub = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            pub_qos_profile
        )

        self.is_connected = False
        self.arming_state = 0
        self.get_logger().info('节点已初始化，正在等待飞控连接...')

    def vehicle_status_callback(self, msg):
        if not self.is_connected:
            self.is_connected = True
            self.get_logger().info('飞控已连接!')
        self.arming_state = msg.arming_state

     # ========== 原有方法（必须保留！） ==========
    def publish_actuator_command(self, actuator_set: int, value: float):
        """
        发送 MAV_CMD_DO_SET_ACTUATOR (187) 到指定 Actuator Set。
        actuator_set: param7, 对应 MAIN 输出的 Actuator Set 编号 (MAIN5=5, MAIN6=6)
        value: param1, 归一化值 -1.0 ~ 1.0, 映射到 PWM min ~ max
        """
        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_ACTUATOR

        # DO_SET_ACTUATOR 参数：
        # param1: actuator 输出值, -1.0(最小PWM) ~ 0.0(中位) ~ 1.0(最大PWM)
        # param7: Actuator Set 编号, 对应 PX4 参数中的 Actuator Set 配置(修改)
        
        #修改
        if actuator_set == 5 or actuator_set == 1:
            msg.param1 = float(value)
            msg.param2 = 0.0
            main_label = "MAIN5"
        elif actuator_set == 6 or actuator_set == 2:
            msg.param1 = 0.0
            msg.param2 = float(value)
            main_label = "MAIN6"
        else:
            self.get_logger().error(f'无效的 actuator_set: {actuator_set}，请使用 1/5 或 2/6')
            return
        
        msg.param3 = 0.0
        msg.param4 = 0.0
        msg.param5 = 0.0
        msg.param6 = 0.0
        msg.param7 = 0.0

        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        self.command_pub.publish(msg)
        self.get_logger().info(
            f'发送舵机指令 -> {main_label}: value={value:.2f}'
        )        
        

   # ========== 新增方法开始 ==========
    def publish_dual_actuator_command(self, value1: float, value2: float):
        """
        同时控制两个舵机（MAIN5 和 MAIN6）
        value1: MAIN5 的值 (-1.0 ~ 1.0)
        value2: MAIN6 的值 (-1.0 ~ 1.0)
        """
        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_ACTUATOR
    
        # 同时设置 param1 (MAIN5) 和 param2 (MAIN6)
        msg.param1 = float(value1)  # MAIN5 → Actuator Set 1
        msg.param2 = float(value2)  # MAIN6 → Actuator Set 2
        msg.param3 = 0.0
        msg.param4 = 0.0
        msg.param5 = 0.0
        msg.param6 = 0.0
        msg.param7 = 0.0
    
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
    
        self.command_pub.publish(msg)
        self.get_logger().info(f'发送双舵机指令 -> MAIN5: {value1:.2f}, MAIN6: {value2:.2f}')
# ========== 新增方法结束 ==========
    def open_servo(self, servo_1=0.0, servo_2=0.0):
        """
        控制两个舵机。
        servo_1: MAIN5 (Actuator Set 5), -1.0 ~ 1.0
        servo_2: MAIN6 (Actuator Set 6), -1.0 ~ 1.0
        """
        self.publish_actuator_command(5, servo_1)   # MAIN5 -> Actuator Set 5
        self.publish_actuator_command(6, servo_2)   # MAIN6 -> Actuator Set 6

