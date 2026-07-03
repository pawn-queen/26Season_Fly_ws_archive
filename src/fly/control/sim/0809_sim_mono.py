#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus, VehicleOdometry
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from collections import deque
import time
from control.DronePositionChecker import DronePositionChecker
from control.AlignmentChecker import AlignmentChecker
from control.ServoControl import ServoControl
from control.visual_servoing import VisualServoingController # 从你的包中导入视觉控制器
import cv2
from enum import Enum
import subprocess
import re
import os
import csv
import argparse # <<< 新增
import sys      # <<< 新增
import numpy as np
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from ament_index_python.packages import get_package_share_directory


class DroppingState(Enum):
    IDLE = 0
    STEP_1_COMMANDED = 1
    STEP_2_COMMANDED = 2
    STEP_3_COMMANDED = 3
    STEP_4_COMMANDED = 4
    COMPLETED = 5

class MissionState(Enum):
    START = 0
    TAKING_OFF = 1
    GLOBAL_SEARCH = 2
    
    TARGETING_CYCLE = 3
    
    TIMEOUT_DROP = 8  # <<< 新增的状态
    # INMISSION = 4

    # === 阶段 2: 侦察任务 ===
    TRANSIT_TO_RECON_OFFBOARD = 9     # 到达侦察区，准备切换回Offboard
    RETURN_TO_CENTER_DROPAREA = 10
    RECON_SEARCH = 12                     # 在侦察区进行视觉搜索
    RECON_CYCLE = 13                      # 按顺序飞到每个侦察点
    
    MISSION_COMPLETE = 14

class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self,args) -> None:
        super().__init__('offboard_control_takeoff_and_land')

        self.show_video = not args.headless  # 如果是headless模式，则不显示视频
        if self.show_video:
            self.get_logger().info("视频显示GUI已启用。")
        else:
            self.get_logger().info("已启用无头模式，将不显示视频GUI。")

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # Create subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v1', self.vehicle_status_callback, qos_profile)
        self.target_position_subscriber = self.create_subscription(Point, '/target_position',
                                                                   self.target_position_callback, qos_profile)
        
        self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.vehicle_odometry_callback, qos_profile)
        
        #这是广角相机的内参和畸变参数
        self.camera_matrix = np.array([
            [465.7411193847656, 0., 320.0],
            [0., 465.7411193847656, 240.0],
            [0., 0., 1.]
        ])
        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) # 假设畸变可以忽略

        self.STATIC_OFFSET_X_FRD = args.depthcam_xoffset # 假设这是旧的x_offset (对应机体前方)
        self.STATIC_OFFSET_Y_FRD = args.depthcam_yoffset  # 假设这是旧的y_offset (对应机体右方)
        
        CAM_POS_IN_BODY = np.array([self.STATIC_OFFSET_X_FRD, self.STATIC_OFFSET_Y_FRD, 0.15])   # 相机位置 (前, 右, 下) in meters
        DROPPER_POS_IN_BODY = np.array([0.0, 0.0, 0.15]) # 投放器位置 (前, 右, 下) in meters

        # --- 2. 定义相机安装姿态的旋转矩阵 ---
        # 这个矩阵代表: 相机X->机体-Y, 相机Y->机体X, 相机Z->机体Z
        R_body_cam = np.array([
            [ 0.,  -1.,  0.],
            [ 1.,  0.,  0.],
            [ 0.,  0.,  1.]
        ])

        # --- 3. 构建从相机到机体的4x4齐次变换矩阵 T_body_cam ---
        self.T_body_cam = np.eye(4)
        self.T_body_cam[:3, :3] = R_body_cam
        self.T_body_cam[:3, 3] = CAM_POS_IN_BODY
        self.get_logger().info("从相机->机体的变换矩阵 T_body_cam 已配置。")

        # --- 4. 定义投放器在机体坐标系下的齐次坐标向量 ---
        self.p_dropper_in_body_h = np.append(DROPPER_POS_IN_BODY, 1)
        self.get_logger().info("投放器相对机体的位置已配置。")
        
        
        # --- 5. 初始化用于存储完整姿态的变量 ---
        self.vehicle_roll = 0.0
        self.vehicle_pitch = 0.0
        # self.init_yaw 将在后面获取，这里无需初始化


        self.get_logger().info("相机内参已配置。")

        # <<< 新增：从参数获取仿真摄像头话题 >>>
        self.declare_parameter('sim_camera_topic', '/camera') # 默认订阅 /camera
        sim_camera_topic = self.get_parameter('sim_camera_topic').get_parameter_value().string_value
        
        
        base_photo_path = args.photo_path
        base_video_path = args.video_path
        
        run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        unique_photo_path = os.path.join(base_photo_path, f"run_{run_timestamp}")
        unique_video_filename = f"mission_{run_timestamp}.avi" # AVI格式与MJPG编码器配合良好        
        
        # === 初始化视觉部分 (带视频录制功能) ===
        self.vision_controller = VisualServoingController(
            model_path=args.model_path,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            # 拍照功能
            enable_photo_capture=False,
            photo_save_path=unique_photo_path, 
            photo_capture_interval=10,
            # <<< 修改：现在由命令行参数控制 >>>
            enable_video_recording=args.record_video, # 设置为 True 来开启录制
            video_save_path=base_video_path,       # 视频保存的目录
            video_filename=unique_video_filename,  # 带有时间戳的唯一文件名
            video_fps=30.0,
            tracking_buffer_size=args.tracking_buffer                         # 视频帧率 (与你的timer频率匹配)
        )
        
        # device_path = self.find_video_device_by_name(args.camera_hint)
        # self.cap = cv2.VideoCapture(device_path if device_path else 0)        
        # if not self.cap.isOpened():
        #     self.get_logger().error("无法打开摄像头！")
        #     rclpy.shutdown()

        self.bridge = CvBridge()
        self.latest_frame = None  # 用于存储最新接收到的图像帧
        self.frame_received_time = self.get_clock().now() # 用于检查图像是否过时
        
        # 创建图像话题订阅者
        self.image_subscriber = self.create_subscription(
            Image,
            sim_camera_topic, # 订阅来自仿真的图像话题
            self.image_callback,
            qos_profile_sensor_data  # 使用 sensor_data QoS 配置
        )
        self.get_logger().info(f"订阅仿真摄像头话题: '{sim_camera_topic}'")

        
        self.is_vision_ready = False

        # === 新增：任务流程管理变量 ===
        self.mission_state = MissionState.START
        self.target_priority = args.target_order 
        self.current_target_index = 0
        self.visited_targets_count = 0
        #=========================================================

        ### --- 新增: 存储计算出的目标世界坐标 --- ###
        self.mission_targets_ned = []  # 格式: [{'name': 'Right', 'coords_ned': (x, y)}, ...]
        self.current_vision_info = [] 


        ### --- 新增: 用于TARGETING_CYCLE状态的内部状态标志 --- ###
        self.is_navigating_to_target = False
        self.is_descending_for_drop = False
        self.is_final_aligning = False

        # ==================== 新增：平滑下降状态变量 ====================
        self.is_smoothing_descent = False      # 是否正在执行平滑下降
        self.smoothing_start_pos = None        # 平滑路径的起点 (x, y, z)
        self.smoothing_end_pos = None          # 平滑路径的终点 (x, y, z)
        self.smoothing_total_steps = 0         # 平滑过程总共需要多少个控制周期
        self.smoothing_step_counter = 0        # 当前执行到第几步

        # <<< 新增：存储动态平滑参数 >>>
        self.smoothing_speed = args.smoothing_speed
        self.min_smoothing_duration = args.min_smoothing_duration
        self.max_smoothing_duration = args.max_smoothing_duration
        self.get_logger().info(f"平滑移动速度配置为: {self.smoothing_speed} m/s "
                            f"(持续时间范围: {self.min_smoothing_duration}s - {self.max_smoothing_duration}s)")

        # ================================================================

        self.is_drop_initiated_for_current_target = False

        self.is_drop_area_calculated = False

        ### 新增: 投放后等待的状态 ###
        self.is_waiting_post_drop = False
        self.post_drop_delay = args.post_drop_delay # 从参数获取
        self.post_drop_start_time = None

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        
        self.target_position = None
        self.last_found_x_NED = None
        self.last_found_y_NED = None
        self.last_found_z_NED = None

        #起飞高度
        self.takeoff_height = args.takeoff_height
        #向前飞行的距离
        self.forward_x = args.forward_x
        # <<< 修改：从命令行参数初始化任务参数 >>>
        self.align_maxstep = args.align_maxstep
        self.afterAlign_descentHeight = args.descent_height
        self.global_search_height = args.search_height

        # <<< 新增：从命令行参数获取超时和延迟设置 >>>
        self.drop_phase_timeout = args.drop_phase_timeout
        self.search_timeout = args.search_timeout
        self.second_align_maxtime = args.second_align_maxtime
        self.first_align_maxtime = args.first_align_maxtime

        self.alignment_altitude_threshold = args.alignment_altitude_threshold


        # <<< 新增：从命令行参数初始化侦察任务参数 >>>
        self.recon_search_height = args.recon_search_height
        self.recon_search_timeout = args.recon_search_timeout
        self.recon_hover_time = args.recon_hover_time
        self.recon_nav_threshold = args.recon_nav_threshold

        self.recon_forward_distance = args.recon_forward_distance


        self.global_search_target_z = None

        self.initial_z = None  # 初始高度
        self.initial_x = None  #
        self.initial_y = None
        self.init_yaw = None

        self.DropArea_x = None
        self.DropArea_y = None
        
        # <<< 新增：用于计时超时的状态变量 >>>
        self.drop_phase_start_time = None
        self.second_align_start_timestamp = None
        self.first_align_start_timestamp = None
        
        self.timeout_drop_start_time = None
        
        
        self.takeoff_target_height = None
        self.is_ReadyToTakeoff = False
        self.is_AtTakeoffHeight = False
        self.is_AtDropArea = False
        self.is_FinishDrop = False

        self.reached_align_height = False


        self.postdrop_waiting_x = None
        self.postdrop_waiting_y = None
        self.postdrop_waiting_z = None

        # 新增日志计数器，用于减少日志输出频率
        self.log_counter = 0
        
        self.timeout_drop_delay = args.timeout_drop_delay

        self.first_alignment_complete = False
        self.second_alignment_complete = False
        self.return_to_recon_center = False
        self.reach_initial_position_above = False

        self.Is_Finish_1st_Drop = False
        self.Is_Finish_2nd_Drop = False

        self.search_start_time = None

        self.last_target_update_time = None
        self.target_timeout_duration = 0.5  # 目标信息超时秒数，例如1秒。可以设为命令行参数。
        
        
        ### 新增: 用于稳定建图的数据收集变量 ###
        self.map_data_collection = []  # 存储多帧的坐标地图

        #==================投水状态机=================
        self.servo_step_delay = args.servo_step_delay  # 每个舵机动作之间的延迟（秒），可以根据实际情况调整
        self.current_dropping_state = {1: DroppingState.IDLE, 2: DroppingState.IDLE}
        self.last_servo_command_time = {1: None, 2: None}

        # ========== 目标像素坐标日志 ==========
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = '~/flylogs'
        log_filename = f'bucket_pixel_log_{timestamp}.csv'
        os.makedirs(log_dir, exist_ok=True)
        self.pixel_log_path = os.path.join(log_dir, log_filename)

        # 打开文件并保留文件句柄和writer对象
        self.pixel_log_file = open(self.pixel_log_path, 'w', newline='', encoding='utf-8')
        self.pixel_log_writer = csv.writer(self.pixel_log_file)
        # 写入表头
        self.pixel_log_writer.writerow(['timestamp', 'target_x', 'target_y', 'bucket_type', 'alignment_stage'])
        self.get_logger().info(f"日志文件已创建并打开: {self.pixel_log_path}")
        # ===========================================================================

        # Create a timer to publish control commands
        self.dt = args.timer_period             # 控制周期 (秒) - 与timer频率一致
        self.control_timer = self.create_timer(self.dt, self.control_timer_callback)
        
        # 创建一个新的、较慢的视觉处理定时器
        self.vision_processing_period = args.vision_timer_period # 10Hz, 可根据设备性能调整
        self.vision_timer = self.create_timer(self.vision_processing_period, self.vision_timer_callback)
        
        # 创建一个线程安全的变量来存储视觉结果
        self.latest_vision_info = []
        self.latest_annotated_frame = None
        # 起飞高度判断阈值
        self.takeoff_threshold = args.takeoff_threshold
        # 向前飞行到达点阈值
        self.nav_threshold = args.nav_threshold
        # 全局搜索到达点阈值
        self.target_approach_threshold = args.target_approach_threshold


        #初始化位置判断器
        self.initPositionChecker = DronePositionChecker(
            logger_func=self.get_logger().info,
            tolerance=0.17, 
            duration=5.0
        )

          # 初始化 AlignmentChecker
        # <<< 修改：使用命令行参数来初始化 AlignmentChecker >>>
        self.first_alignment_checker = AlignmentChecker(
            logger_func=self.get_logger().info,
            threshold=args.first_align_threshold,
            time_window=args.first_align_time_window,
            check_frequency=args.first_align_check_freq
        )
        self.second_alignment_checker = AlignmentChecker(
            logger_func=self.get_logger().info,
            threshold=args.second_align_threshold,
            time_window=args.second_align_time_window,
            check_frequency=args.second_align_check_freq
        )
        # 初始化舵机控制器
        self.servo_control = ServoControl()
        
        # ========== PID控制参数设置区域 ==========
        # 📌 饱和P控制参数（大误差阶段）
        
        # 📌 细调阶段PID参数（小误差阶段）
        self.epsilon = self.align_maxstep  # 切换阈值 (0.2m) - 可调参数
        self.Kp_fine = args.kp  # P增益 - 可调参数 (建议范围: 1.0-2.5)
        self.Ki = args.ki       # I增益 - 可调参数 (建议范围: 0.1-0.8)
        self.Kd = args.kd
        self.Kf = args.kf
        
        # 📌 PID状态变量
        self.integral_x = 0.0      # X方向积分项
        self.integral_y = 0.0      # Y方向积分项
        self.last_error_x = 0.0    # 上次X误差 (用于微分计算)
        self.last_error_y = 0.0    # 上次Y误差 (用于微分计算)
        
        
        # 📌 积分限幅参数
        self.max_integral = self.epsilon  # 积分限幅值 - 可调参数
        # =========================================


        # === 新增：侦察任务相关变量 ===
        self.recon_targets_ned = []              # 存储5个侦察目标的NED坐标
        self.current_recon_index = 0             # 当前正在飞往的侦察目标索引
        self.is_recon_map_built = False          # 侦察地图是否已建立
        self.recon_search_start_time = None      # 侦察搜索开始时间
        self.recon_hover_start_time = None       # 到达侦察点后，悬停开始时间
        self.is_hovering_at_recon_point = False  # 是否正在悬停侦察的标志



    # +++ (新增的回调函数) +++
    def image_callback(self, msg: Image):
        """
        接收来自仿真摄像头的图像消息，并将其转换为OpenCV格式。
        """
        try:
            # 将 ROS Image 消息转换为 OpenCV 图像 (bgr8 是标准彩色格式)
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.frame_received_time = self.get_clock().now()
        except Exception as e:
            self.get_logger().error(f"无法转换图像: {e}")
            



    def target_position_callback(self, msg: Point):
        """Callback function for receiving target position."""
        self.target_position = msg
         # <<<更新收到目标的时间戳 >>>
        self.last_target_update_time = self.get_clock().now()  

    def fly_to_position(self, x, y, z):
        """Fly to the specified position."""
        self.publish_position_setpoint(x, y, z)

    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        self.vehicle_local_position = vehicle_local_position

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status

    def vehicle_odometry_callback(self, msg: VehicleOdometry):
        """Callback to get the drone's full attitude (roll, pitch, yaw)."""
        # PX4 odometry msg.q is [w, x, y, z]
        # Scipy Rotation needs [x, y, z, w]
        q = [msg.q[1], msg.q[2], msg.q[3], msg.q[0]]
        
        # 从四元数转换为欧拉角 (roll, pitch, yaw)，单位是弧度
        (self.vehicle_roll, 
         self.vehicle_pitch, 
         _) = R.from_quat(q).as_euler('xyz', degrees=False)
        # Yaw我们继续使用更稳定的 vehicle_local_position.heading


    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        # self.get_logger().info("Switching to offboard mode")

    def start_mission(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=4.0, param2=3.0)
        self.get_logger().info("Switching to Mission mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def return_to_launch(self):
        """Switch to RTL mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH)
        self.get_logger().info("Switching to Return-to-Launch (RTL) mode")

    def publish_offboard_control_heartbeat_signal(self):
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_position_setpoint(self, x: float, y: float, z: float):
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        if self.init_yaw is None:
            msg.yaw = 0.00
        else:
            msg.yaw = self.init_yaw  # (90 degree)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

        # <<< 新增：重写 destroy_node 方法以进行清理 >>>
    def destroy_node(self):
        """在节点关闭前，执行必要的清理工作。"""
        self.get_logger().info("节点正在关闭，执行清理程序...")
        # 清理视觉控制器（保存视频）
        if self.vision_controller:
            self.vision_controller.cleanup()
        # 关闭日志文件
        if hasattr(self, 'pixel_log_file') and not self.pixel_log_file.closed:
            self.pixel_log_file.close()
            self.get_logger().info("像素日志文件已关闭。")
        # # 清理摄像头
        # if self.cap and self.cap.isOpened():
        #     self.cap.release()
        # 关闭所有OpenCV窗口
        if self.show_video:
            cv2.destroyAllWindows()
        # 调用父类的方法完成ROS节点的销毁
        super().destroy_node()
        self.get_logger().info("清理完成，节点已关闭。")
    
    # def find_video_device_by_name(self,name_hint="USB Camera"):
    # # (This function remains unchanged)
    #     try:
    #         result = subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True, text=True, check=True)
    #     except (FileNotFoundError, subprocess.CalledProcessError): return None
    #     lines = result.stdout.splitlines()
    #     matched_device_name = False
    #     for line in lines:
    #         if name_hint in line: matched_device_name = True
    #         elif matched_device_name and "/dev/video" in line:
    #             match = re.search(r"(/dev/video\d+)", line)
    #             if match: return match.group(1)
    #     return None
 
    
    def drop_payload(self, drop_number: int):
        """
        启动指定编号的多步骤投水序列。
        这个函数只负责启动，不负责管理过程。
        """
        if self.current_dropping_state[drop_number] == DroppingState.IDLE:
            self.get_logger().info(f"启动第 {drop_number} 次投水序列...")
            self.get_logger().info(f"第 {drop_number} 次投水 - 步骤 1: (0, 0)")
            if drop_number == 1 :
                self.servo_control.open_servo(0.0, 1.0)
            elif drop_number ==2 :
                self.servo_control.open_servo(0.0, -1.0)
            self.current_dropping_state[drop_number] = DroppingState.STEP_1_COMMANDED
            # 使用ROS 2的时钟
            self.last_servo_command_time[drop_number] = self.get_clock().now()

    def manage_dropping_sequence(self, drop_number: int) -> bool:
        """
        非阻塞地管理投水过程，应该在 timer_callback 中被反复调用。
        返回: True 如果序列完成，否则 False。
        """
        state = self.current_dropping_state[drop_number]
        
        if state == DroppingState.IDLE:
            return False
        if state == DroppingState.COMPLETED:
            return True

        elapsed_time = (self.get_clock().now() - self.last_servo_command_time[drop_number]).nanoseconds / 1e9
        if elapsed_time < self.servo_step_delay:
            return False

        self.get_logger().info(f"第 {drop_number} 次投水 - 执行下一步...")

        # 这里使用您在ServoTester中验证过的舵机指令
        if state == DroppingState.STEP_1_COMMANDED:
            if drop_number == 1:
                self.servo_control.open_servo(0.0, 1.0)
            else: # drop_number == 2
                self.servo_control.open_servo(0.0, -1.0)
            self.current_dropping_state[drop_number] = DroppingState.STEP_2_COMMANDED
            self.last_servo_command_time[drop_number] = self.get_clock().now()
        
        elif state == DroppingState.STEP_2_COMMANDED:
            self.servo_control.open_servo(0.0, 0.0)
            self.current_dropping_state[drop_number] = DroppingState.STEP_3_COMMANDED
            self.last_servo_command_time[drop_number] = self.get_clock().now()

        elif state == DroppingState.STEP_3_COMMANDED:
            if drop_number == 1:
                self.servo_control.open_servo(1.0, 0.0)
            else: # drop_number == 2
                self.servo_control.open_servo(-1.0, 0.0)
            self.current_dropping_state[drop_number] = DroppingState.STEP_4_COMMANDED
            self.last_servo_command_time[drop_number] = self.get_clock().now()
            
        elif state == DroppingState.STEP_4_COMMANDED:
            self.servo_control.open_servo(0.0, 0.0)
            self.get_logger().info(f"第 {drop_number} 次投水序列完成。")
            self.current_dropping_state[drop_number] = DroppingState.COMPLETED
            return True
            
        return False

    def takeoff_relative(self): # 不再需要 relative_height 参数
        """
        飞向预先计算好的目标起飞高度。
        这个函数假定 self.takeoff_target_height 和 self.init_yaw 等已经被设置。
        """
        if self.takeoff_target_height is None:
            self.get_logger().error("takeoff_relative 被调用，但目标起飞高度未设置！")
            return
        
        # 直接命令无人机飞到（初始x, 初始y, 目标z）
        # fly_to_position_FRD2NED 会自动使用 self.initial_x, self.initial_y, self.init_yaw
        self.fly_to_position_FRD2NED(0.0, 0.0, self.takeoff_target_height)

    def takeoff_height_check(self):
        """
        检查是否到达相对目标高度
        :param threshold: 高度误差阈值
        :return: True 如果到达目标高度，否则 False
        """
        if self.takeoff_target_height is None:
            self.get_logger().warn("目标高度尚未设置！")
            return False
        current_height = self.vehicle_local_position.z
        height_error = abs(current_height - self.takeoff_target_height)
        # 为了减少日志输出，只有每隔一定周期时才打印此日志
        if self.log_counter % 25 == 0:
            self.get_logger().info(f"当前高度：{current_height:.2f} 米，目标高度：{self.takeoff_target_height:.2f} 米，高度误差：{height_error:.2f} 米")
        if height_error < self.takeoff_threshold:
            self.is_AtTakeoffHeight = True

    def calculate_drop_area_once(self, x):
        """
        仅计算一次投水区的NED坐标并存储。
        这个函数只在状态切换时被调用一次。
        """
        # 使用 coordinate_FRD2NED 函数计算目标点，但不发布
        self.DropArea_x, self.DropArea_y = self.coordinate_FRD2NED(x, 0)
        self.get_logger().info(f"投水区目标点已计算 (NED): x={self.DropArea_x:.2f}, y={self.DropArea_y:.2f}")

    def navigate_to_drop_area(self):
        """
        在每个循环中导航至投水区并检查是否到达。
        这是一个闭环控制函数。
        """
        # 1. 持续发布飞向预定目标点的指令
        # 目标高度保持在起飞高度
        self.publish_position_setpoint(self.DropArea_x, self.DropArea_y, self.takeoff_target_height)

        # 2. 检查是否已经到达
        current_x = self.vehicle_local_position.x
        current_y = self.vehicle_local_position.y
        error = math.sqrt((current_x - self.DropArea_x)**2 + (current_y - self.DropArea_y)**2)

        if self.log_counter % 25 == 0:
            self.get_logger().info(f"导航至投水区... "
                                   f"当前:({current_x:.2f}, {current_y:.2f}), "
                                   f"目标:({self.DropArea_x:.2f}, {self.DropArea_y:.2f}), "
                                   f"距离误差: {error:.2f} m")

        if error < self.nav_threshold:
            self.is_AtDropArea = True
            self.get_logger().info("已到达投水区！")
    
    def first_alignment_check(self, target_x, target_y):
        """Check first alignment with the target."""
        current_x = self.vehicle_local_position.x
        current_y = self.vehicle_local_position.y
        is_align_now = self.first_alignment_checker.check(
            current_x,
            current_y,
            target_x=target_x,
            target_y=target_y
)       
        if is_align_now:
            self.first_alignment_complete = True
            self.second_alignment_checker.reset()
            self.get_logger().info("------------------------first对准完成！------------------------")

    def second_alignment_check(self, target_x, target_y):
        """Check second alignment with the target."""
        is_align_now = self.second_alignment_checker.check(
    current_x=self.vehicle_local_position.x,
    current_y=self.vehicle_local_position.y,
            target_x=target_x,
            target_y=target_y
        )
        if is_align_now:
            self.second_alignment_complete = True
            self.get_logger().info("-------------------------second对准完成！------------------------")

    def fly_to_position_FRD2NED(self,x,y,z):
        '''
        通过旋转矩阵, 将FRD坐标系转换为NED坐标系。再根据初始误差增加平移矩阵。

        '''
        x_target = x*math.cos(self.init_yaw)-y*math.sin(self.init_yaw) + self.initial_x
        y_target = x*math.sin(self.init_yaw)+y*math.cos(self.init_yaw) + self.initial_y
        z_target = z
        self.publish_position_setpoint(x_target, y_target, z_target)
        return x_target, y_target

    def coordinate_NED2FRD(self,x_NED,y_NED):
        '''
        将NED坐标转换为FRD坐标。
        '''
        x_FRD = (x_NED-self.initial_x)*math.cos(self.init_yaw)+(y_NED-self.initial_y)*math.sin(self.init_yaw)
        y_FRD = -(x_NED-self.initial_x)*math.sin(self.init_yaw)+(y_NED-self.initial_y)*math.cos(self.init_yaw)
        return x_FRD, y_FRD
    
    def coordinate_NED2FRD_vector(self, vec_ned_x, vec_ned_y):
        '''
        将NED坐标系下的2D向量，仅通过旋转，转换为FRD机体坐标系下的2D向量。
        '''
        current_yaw = self.vehicle_local_position.heading
        # 向量变换只涉及旋转，不涉及平移
        vec_frd_x = vec_ned_x * math.cos(current_yaw) + vec_ned_y * math.sin(current_yaw)
        vec_frd_y = -vec_ned_x * math.sin(current_yaw) + vec_ned_y * math.cos(current_yaw)
        return vec_frd_x, vec_frd_y

    def coordinate_FRD2NED(self,x,y):
        '''
        将FRD坐标转换为NED坐标。
        '''
        x_target = x*math.cos(self.init_yaw)-y*math.sin(self.init_yaw) + self.initial_x
        y_target = x*math.sin(self.init_yaw)+y*math.cos(self.init_yaw) + self.initial_y

        return x_target, y_target
    
    def reset_for_next_target(self):
        """为下一个目标重置所有相关的状态标志"""
        self.get_logger().info("重置状态以准备下一个目标...")
        self.first_alignment_complete = False
        self.second_alignment_complete = False
        self.first_alignment_checker.reset()
        self.second_alignment_checker.reset()
        self.second_align_start_timestamp = None
        self.first_align_start_timestamp = None
        self.target_position = None
        self.last_found_x_NED = None
        self.last_found_y_NED = None
        self.last_found_z_NED = None
        
        # 重置TARGETING_CYCLE的内部状态
        self.is_navigating_to_target = False
        self.is_descending_for_drop = False
        self.is_final_aligning = False

        self.is_drop_initiated_for_current_target = False
        
        # 增加投放计数和索引
        self.visited_targets_count += 1
        self.current_target_index += 1
        ### --- 新增的关键代码 --- ###
        # 重新启动下一个目标的导航流程
        self.is_navigating_to_target = True
        self.get_logger().info("状态机已重置，开始导航至下一个目标。")

    def _start_smooth_move(self, end_pos_ned: tuple):
        """
        计算并启动到目标点的动态平滑移动。
        """
        # 1. 设置起点为当前无人机的位置
        start_pos_ned = (
            self.vehicle_local_position.x,
            self.vehicle_local_position.y,
            self.vehicle_local_position.z
        )
        self.smoothing_start_pos = start_pos_ned
        self.smoothing_end_pos = end_pos_ned

        # 2. 计算三维空间距离
        dx = end_pos_ned[0] - start_pos_ned[0]
        dy = end_pos_ned[1] - start_pos_ned[1]
        dz = end_pos_ned[2] - start_pos_ned[2]
        distance = math.sqrt(dx**2 + dy**2 + dz**2)

        # 3. 根据速度计算理想持续时间
        if self.smoothing_speed > 0.01: # 避免除以零
            ideal_duration = distance / self.smoothing_speed
        else:
            ideal_duration = self.max_smoothing_duration

        # 4. 将持续时间限制在预设的最小和最大值之间
        clamped_duration = max(self.min_smoothing_duration, min(ideal_duration, self.max_smoothing_duration))
        
        # 5. 根据最终持续时间计算总步数
        self.smoothing_total_steps = int(clamped_duration / self.dt)
        if self.smoothing_total_steps < 1:
            self.smoothing_total_steps = 1 # 确保至少有一步

        self.get_logger().info(f"启动平滑移动: 距离={distance:.2f}m, "
                               f"计算耗时={clamped_duration:.2f}s, "
                               f"总步数={self.smoothing_total_steps}")
        
        # 6. 重置计数器并激活平滑移动标志
        self.smoothing_step_counter = 0
        self.is_smoothing_descent = True # 使用相同的标志位

    def adjust_to_target(self):
        """Adjust drone position towards the current target."""   
        # <<< 新增：超时检查逻辑 >>>
        is_target_valid = False
        if self.target_position and self.last_target_update_time:
            elapsed_time = (self.get_clock().now() - self.last_target_update_time).nanoseconds / 1e9
            if elapsed_time < self.target_timeout_duration:
                is_target_valid = True
            else:
                if self.log_counter % 25 == 0:
                    self.get_logger().warn(f"目标信息已超时 ({elapsed_time:.2f}s > {self.target_timeout_duration}s)，将忽略旧目标。")
                    
        is_in_second_alignment = self.first_alignment_complete and not self.second_alignment_complete
        is_in_first_alignment = not self.first_alignment_complete

        if is_in_first_alignment:
            # 启动计时器 (如果尚未启动)
            if self.first_align_start_timestamp is None:
                self.first_align_start_timestamp = self.get_clock().now()
                self.get_logger().info(f"第一次对准开始，启动 {self.first_align_maxtime} 秒超时计时器。")

            # 计算已过时间
            elapsed_first_align_time = (self.get_clock().now() - self.first_align_start_timestamp).nanoseconds / 1e9

            # 检查是否超时
            if elapsed_first_align_time > self.first_align_maxtime:
                if self.current_dropping_state[1] == DroppingState.IDLE and not self.Is_Finish_1st_Drop:
                    self.drop_payload(1) # 启动第一次投水
                    self.is_drop_initiated_for_current_target = True
                    self.second_align_start_timestamp = None

                    self.postdrop_waiting_x = self.vehicle_local_position.x
                    self.postdrop_waiting_y = self.vehicle_local_position.y
                    self.postdrop_waiting_z = self.vehicle_local_position.z

                    self.get_logger().warn(f"进行目标1超时投放！")
                    self.get_logger().info(f"将在原地悬停 {self.post_drop_delay} 秒...")
                    
                    ### MODIFIED ###
                    # 进入投放后等待状态，而不是直接重置
                    self.is_final_aligning = False
                    self.is_waiting_post_drop = True
                    self.post_drop_start_time = self.get_clock().now()
                    return

                
                elif self.current_dropping_state[2] == DroppingState.IDLE and self.Is_Finish_1st_Drop and not self.Is_Finish_2nd_Drop:
                    self.drop_payload(2) # 启动第二次投水
                    self.is_drop_initiated_for_current_target = True
                    self.second_align_start_timestamp = None

                    self.postdrop_waiting_x = self.vehicle_local_position.x
                    self.postdrop_waiting_y = self.vehicle_local_position.y
                    self.postdrop_waiting_z = self.vehicle_local_position.z

                    self.get_logger().warn(f"进行目标2超时投放！")
                    self.get_logger().info(f"将在原地悬停 {self.post_drop_delay} 秒...")
                    
                    ### MODIFIED ###
                    # 进入投放后等待状态，而不是直接重置
                    self.is_final_aligning = False
                    self.is_waiting_post_drop = True
                    self.post_drop_start_time = self.get_clock().now()

                    return
        
        # 如果正处于第二次对准阶段，无条件检查超时
        if is_in_second_alignment:
            # 启动计时器 (如果尚未启动)
            if self.second_align_start_timestamp is None:
                self.second_align_start_timestamp = self.get_clock().now()
                self.get_logger().info(f"第二次对准开始，启动 {self.second_align_maxtime} 秒超时计时器。")

            # 计算已过时间
            elapsed_drop_time = (self.get_clock().now() - self.second_align_start_timestamp).nanoseconds / 1e9
            
            # 检查是否超时
            if elapsed_drop_time > self.second_align_maxtime:
                if self.current_dropping_state[1] == DroppingState.IDLE and not self.Is_Finish_1st_Drop:
                    self.drop_payload(1) # 启动第一次投水
                    self.is_drop_initiated_for_current_target = True
                    self.second_align_start_timestamp = None

                    self.postdrop_waiting_x = self.vehicle_local_position.x
                    self.postdrop_waiting_y = self.vehicle_local_position.y
                    self.postdrop_waiting_z = self.vehicle_local_position.z

                    self.get_logger().warn(f"进行目标1超时投放！")
                    self.get_logger().info(f"将在原地悬停 {self.post_drop_delay} 秒...")
                    
                    ### MODIFIED ###
                    # 进入投放后等待状态，而不是直接重置
                    self.is_final_aligning = False
                    self.is_waiting_post_drop = True
                    self.post_drop_start_time = self.get_clock().now()
                    return

                
                elif self.current_dropping_state[2] == DroppingState.IDLE and self.Is_Finish_1st_Drop and not self.Is_Finish_2nd_Drop:
                    self.drop_payload(2) # 启动第二次投水
                    self.is_drop_initiated_for_current_target = True
                    self.second_align_start_timestamp = None

                    self.postdrop_waiting_x = self.vehicle_local_position.x
                    self.postdrop_waiting_y = self.vehicle_local_position.y
                    self.postdrop_waiting_z = self.vehicle_local_position.z

                    self.get_logger().warn(f"进行目标2超时投放！")
                    self.get_logger().info(f"将在原地悬停 {self.post_drop_delay} 秒...")
                    
                    ### MODIFIED ###
                    # 进入投放后等待状态，而不是直接重置
                    self.is_final_aligning = False
                    self.is_waiting_post_drop = True
                    self.post_drop_start_time = self.get_clock().now()

                    return
            
      
        if is_target_valid:
            # ========== pid控制实现，记录目标像素坐标 ========== 
            self.pixel_log_writer.writerow([
        time.time(),
        self.target_position.x,
        self.target_position.y,
        # 'bucket_type' 和 'alignment_stage' 你可以根据当前状态添加
    ])
            P_cam_h = np.array([self.target_position.x, self.target_position.y, self.target_position.z, 1])
            
            # (A) 相机 -> 机体
            P_target_in_body_h = self.T_body_cam @ P_cam_h

            # (B) 构建动态的 世界 -> 机体 变换矩阵
            current_yaw = self.vehicle_local_position.heading
            R_world_body = R.from_euler('xyz', [self.vehicle_roll, self.vehicle_pitch, current_yaw]).as_matrix()
            T_world_body = np.eye(4)
            T_world_body[:3, :3] = R_world_body
            T_world_body[:3, 3] = [self.vehicle_local_position.x, self.vehicle_local_position.y, self.vehicle_local_position.z]
            
            # (C) 机体 -> 世界，得到目标的真实世界坐标
            P_target_in_world_h = T_world_body @ P_target_in_body_h

            # (D) 计算投放器在世界坐标系下的绝对位置
            p_dropper_in_world_h = T_world_body @ self.p_dropper_in_body_h

            # (E) 计算控制误差：目标的真实世界位置 - 投放器的真实世界位置
            error_ned = P_target_in_world_h[:3] - p_dropper_in_world_h[:3]
            
            # (F) 将NED世界误差向量，转换为FRD机体误差向量，以输入给PID
            error_frd_x, error_frd_y = self.coordinate_NED2FRD_vector(error_ned[0], error_ned[1])


            if self.log_counter % 25 == 0: # 大约每秒打印一次，避免刷屏
            # 1. 计算真实的动态世界偏移量 (NED)
            #    这是投放器的世界位置 减去 无人机中心的世界位置
                p_camera_in_body_h = np.array([0, 0, 0, 1]) # 相机坐标系的原点
                p_camera_in_world_h = T_world_body @ (self.T_body_cam @ p_camera_in_body_h)
                
                # --- 2. 计算从“相机”到“投放器”的“总偏移”向量 (在世界坐标系下) ---
                total_offset_ned = p_dropper_in_world_h[:3] - p_camera_in_world_h[:3]
                
                total_offset_ned_D_to_C = -total_offset_ned

                # --- 3. 将这个用于比较的 D->C 向量转换回机体坐标系 ---
                comparison_offset_frd_x, comparison_offset_frd_y = self.coordinate_NED2FRD_vector(
                    total_offset_ned_D_to_C[0], total_offset_ned_D_to_C[1]
                )

                # # --- 4. 打印清晰的、定义一致的对比日志 ---
                # self.get_logger().info("--- [补偿向量对比 (机体坐标系 FRD)] ---")
                # self.get_logger().info(f"  [静态补偿值]: Forward={self.STATIC_OFFSET_X_FRD:.4f}, Right={self.STATIC_OFFSET_Y_FRD:.4f}")
                # self.get_logger().info(f"  [动态补偿值]: Forward={comparison_offset_frd_x:.4f}, Right={comparison_offset_frd_y:.4f} (Roll:{math.degrees(self.vehicle_roll):.1f}°, Pitch:{math.degrees(self.vehicle_pitch):.1f}°)")
                # self.get_logger().info("-------------------------------------------")

            # === 2. 将精确误差 "喂" 给你的PID控制器 ===
            distance = math.hypot(error_frd_x, error_frd_y)
            
            if distance < self.epsilon:
                # ——— PID细调阶段 (使用新的精确误差) ———
                if self.log_counter % 25 == 0: self.get_logger().info(f"PID细调阶段 - 精确误差:{distance:.3f}m")
                error_x = error_frd_x
                error_y = error_frd_y
                # ... (你的PIDF计算逻辑完全不变) ...
                self.integral_x += error_x * self.dt
                self.integral_y += error_y * self.dt
                # 积分限幅
                self.integral_x = max(min(self.integral_x, self.max_integral), -self.max_integral)
                self.integral_y = max(min(self.integral_y, self.max_integral), -self.max_integral)
                
                # 📌 微分项计算
                derivative_x = (error_x - self.last_error_x) / self.dt
                derivative_y = (error_y - self.last_error_y) / self.dt
                velocity_x_ned = self.vehicle_local_position.vx
                velocity_y_ned = self.vehicle_local_position.vy
                vel_x_body_frame, vel_y_body_frame = self.coordinate_NED2FRD_vector(velocity_x_ned, velocity_y_ned)
                feedforward_x = self.Kf * vel_x_body_frame
                feedforward_y = self.Kf * vel_y_body_frame
                control_x = (self.Kp_fine * error_x + self.Ki * self.integral_x + self.Kd * derivative_x - feedforward_x)
                control_y = (self.Kp_fine * error_y + self.Ki * self.integral_y + self.Kd * derivative_y - feedforward_y)
                self.last_error_x = error_x
                self.last_error_y = error_y
                
                if self.log_counter % 25 == 0:
                    p_term = self.Kp_fine * error_x
                    i_term = self.Ki * self.integral_x
                    d_term = self.Kd * derivative_x
                    f_term = -feedforward_x
                    self.get_logger().info(f"PIDF输出: P={p_term:.3f}, I={i_term:.3f}, D={d_term:.3f}, F={f_term:.3f}")
            else:
                # ——————— 大误差阶段：饱和P控制 ———————
                if self.log_counter % 25 == 0:
                    self.get_logger().info(f"饱和P控制阶段 - 误差:{distance:.3f}m >= 阈值:{self.epsilon:.3f}m")
                
                # 📌 饱和比例控制
                scale = self.align_maxstep / distance
                control_x = error_frd_x * scale
                control_y = error_frd_y * scale
                self.integral_x, self.integral_y, self.last_error_x, self.last_error_y = 0.0, 0.0, 0.0, 0.0

            # === 3. [不变部分] 计算并发布目标点 ===
            current_x_frd, current_y_frd = self.coordinate_NED2FRD(self.vehicle_local_position.x, self.vehicle_local_position.y)
            target_x_FRD = current_x_frd + control_x
            target_y_FRD = current_y_frd + control_y
            target_x_NED, target_y_NED = self.coordinate_FRD2NED(target_x_FRD, target_y_FRD)
            
            # === 4. [修改部分] 计算用于对准检查的精确目标点 ===
            # 检查点 = 无人机当前位置 + NED误差向量 (即我们希望无人机飞到的位置)
            precise_target_x_NED = self.vehicle_local_position.x + error_ned[0]
            precise_target_y_NED = self.vehicle_local_position.y + error_ned[1]
            
            # ============== 两次对准逻辑 ==============
            # First alignment
            if not self.first_alignment_complete:
                if self.log_counter % 25 == 0:
                    self.get_logger().info("执行第一次对准")
                self.fly_to_position(target_x_NED, target_y_NED, self.takeoff_target_height)
                self.first_alignment_check(precise_target_x_NED, precise_target_y_NED)
                self.last_found_x_NED = self.vehicle_local_position.x
                self.last_found_y_NED = self.vehicle_local_position.y
                self.last_found_z_NED = self.takeoff_target_height

            elif self.first_alignment_complete and not self.second_alignment_complete:
                if self.log_counter % 25 == 0:
                    self.get_logger().info("执行第二次精确对准")
                self.fly_to_position(target_x_NED, target_y_NED, self.takeoff_target_height + self.afterAlign_descentHeight)

                # <<< 新增：高度门控 >>>
                current_z = self.vehicle_local_position.z
                target_z = self.takeoff_target_height + self.afterAlign_descentHeight
                altitude_error = abs(current_z - target_z)

                if not self.reached_align_height:
                    if altitude_error < self.alignment_altitude_threshold:
                        self.reached_align_height = True
                        self.get_logger().warn(f"高度达到，检查对准精度")
                else:
                    self.second_alignment_check(precise_target_x_NED, precise_target_y_NED)        

                self.last_found_x_NED = self.vehicle_local_position.x
                self.last_found_y_NED = self.vehicle_local_position.y
                self.last_found_z_NED = self.takeoff_target_height + self.afterAlign_descentHeight
            
            # ============== 投水逻辑 ==============
            if self.first_alignment_complete and self.second_alignment_complete and not self.is_drop_initiated_for_current_target:
    # 只负责启动，不设置完成标志
                if self.current_dropping_state[1] == DroppingState.IDLE and not self.Is_Finish_1st_Drop:
                    self.drop_payload(1) # 启动第一次投水
                    self.is_drop_initiated_for_current_target = True
                    self.second_align_start_timestamp = None

                    self.postdrop_waiting_x = self.vehicle_local_position.x
                    self.postdrop_waiting_y = self.vehicle_local_position.y
                    self.postdrop_waiting_z = self.vehicle_local_position.z

                    self.get_logger().info(f"进行目标1投放！")
                    self.get_logger().info(f"将在原地悬停 {self.post_drop_delay} 秒...")
                    
                    ### MODIFIED ###
                    # 进入投放后等待状态，而不是直接重置
                    self.reached_align_height = False
                    self.is_final_aligning = False
                    self.is_waiting_post_drop = True
                    self.post_drop_start_time = self.get_clock().now()
                    return


                
                elif self.current_dropping_state[2] == DroppingState.IDLE and self.Is_Finish_1st_Drop and not self.Is_Finish_2nd_Drop:
                    self.drop_payload(2) # 启动第二次投水
                    self.is_drop_initiated_for_current_target = True
                    self.second_align_start_timestamp = None

                    self.postdrop_waiting_x = self.vehicle_local_position.x
                    self.postdrop_waiting_y = self.vehicle_local_position.y
                    self.postdrop_waiting_z = self.vehicle_local_position.z

                    self.get_logger().info(f"进行目标2投放！")
                    self.get_logger().info(f"将在原地悬停 {self.post_drop_delay} 秒...")
                    
                    ### MODIFIED ###
                    # 进入投放后等待状态，而不是直接重置
                    self.reached_align_height = False
                    self.is_final_aligning = False
                    self.is_waiting_post_drop = True
                    self.post_drop_start_time = self.get_clock().now()

                    return


                
        else:
            # ============== 无目标时的处理 ==============
            if self.last_found_x_NED and self.last_found_y_NED and self.last_found_z_NED:
                if self.log_counter % 25 == 0:
                    self.get_logger().info("无新目标，使用上次记录位置")
                self.fly_to_position(self.last_found_x_NED, self.last_found_y_NED, self.last_found_z_NED)
            else:
                if self.log_counter % 25 == 0:
                    self.get_logger().info("无目标记录，原地等待")
                self.fly_to_position(self.vehicle_local_position.x, self.vehicle_local_position.y, self.takeoff_target_height)
    
    
    def _calculate_and_store_average_map(self):
        """
        计算收集到的多帧地图数据的平均值，并将其存储到最终的NED坐标地图中。
        """
        if not self.map_data_collection:
            self.get_logger().error("无法计算平均地图，因为没有收集到数据。")
            return

        # 初始化用于求和的字典
        sum_coords = {"Left": [0.0, 0.0], "Middle": [0.0, 0.0], "Right": [0.0, 0.0]}
        counts = {"Left": 0, "Middle": 0, "Right": 0}

        # 累加所有收集到的坐标
        for frame_map in self.map_data_collection:
            for name, coords in frame_map.items():
                if name in sum_coords:
                    sum_coords[name][0] += coords[0] # x_frd
                    sum_coords[name][1] += coords[1] # y_frd
                    counts[name] += 1
        
        # 计算平均值并转换为NED坐标
        for name in sum_coords.keys():
            if counts[name] > 0:
                avg_x_frd = sum_coords[name][0] / counts[name]
                avg_y_frd = sum_coords[name][1] / counts[name]

                # 转换为相对于无人机初始位置的绝对FRD坐标
                abs_x_frd = self.forward_x + avg_x_frd
                abs_y_frd = 0.0 + avg_y_frd
                
                # 转换为NED坐标并存储
                ned_x, ned_y = self.coordinate_FRD2NED(abs_x_frd, abs_y_frd)
                self.world_target_coordinates_ned[name] = (ned_x, ned_y)
                self.get_logger().info(f"  -> 平均坐标 '{name}' (FRD): ({avg_x_frd:.2f}, {avg_y_frd:.2f}) -> (NED): ({ned_x:.2f}, {ned_y:.2f})")

    def _build_final_mission_map(self, named_targets_frd):
        """
        根据视觉模块返回的命名目标列表和用户指定的优先级，构建最终任务地图。
        """
        self.mission_targets_ned.clear()
        if not named_targets_frd:
            self.get_logger().warn("建图失败：视觉模块未确认任何目标。")
            return

        # 将视觉结果转换为一个字典，方便按名称查找: {'Left': {...}, 'Middle': {...}}
        vision_map = {target['name']: target for target in named_targets_frd}
        
        self.get_logger().info(f"建图开始... 视觉系统发现: {list(vision_map.keys())}")
        self.get_logger().info(f"将按照用户指定的顺序进行打击: {self.target_priority}")

        # 按照用户指定的优先级列表来构建任务
        for target_name in self.target_priority:
            if target_name in vision_map:
                target_data = vision_map[target_name]
                x_frd, y_frd = target_data['coords_frd']
                
                abs_x_frd = self.forward_x + x_frd
                abs_y_frd = 0.0 + y_frd

                ned_x, ned_y = self.coordinate_FRD2NED(abs_x_frd, abs_y_frd)
                
                self.mission_targets_ned.append({
                    'name': target_name,
                    'coords_ned': (ned_x, ned_y)
                })
                self.get_logger().info(f"  -> 已规划目标 '{target_name}' @ NED({ned_x:.2f}, {ned_y:.2f})")
            else:
                self.get_logger().warn(f"  -> 用户指定的目标 '{target_name}' 未在视野中被确认，将跳过。")
        
        self.get_logger().info("最终任务地图构建完成。")
    

    def _build_recon_mission_map(self, named_targets_frd):
        """
        为侦察阶段构建任务地图。
        """
        self.recon_targets_ned.clear()
        if not named_targets_frd:
            self.get_logger().warn("侦察建图失败：视觉模块未确认任何目标。")
            return

        self.get_logger().info("开始构建侦察任务地图...")
        # 侦察任务不需要用户指定顺序，直接按视觉模块返回的顺序（通常是按x轴排序）
        for target_data in named_targets_frd:
            # 这里我们假设无人机在切换回Offboard后位置变化不大
            # 一个更鲁棒的方法是记录切换回Offboard时的精确位置
            current_x, current_y = self.coordinate_NED2FRD(self.vehicle_local_position.x, self.vehicle_local_position.y)

            x_frd, y_frd = target_data['coords_frd']
            
            # 目标的绝对FRD坐标 = 当前无人机FRD坐标 + 目标相对无人机的FRD坐标
            abs_x_frd = current_x + x_frd
            abs_y_frd = current_y + y_frd

            ned_x, ned_y = self.coordinate_FRD2NED(abs_x_frd, abs_y_frd)
            
            self.recon_targets_ned.append({
                'name': target_data['name'],
                'coords_ned': (ned_x, ned_y)
            })
            self.get_logger().info(f"  -> 已规划侦察目标 '{target_data['name']}' @ NED({ned_x:.2f}, {ned_y:.2f})")
        
        self.get_logger().info("侦察任务地图构建完成。")


    #定时器
    def control_timer_callback(self) -> None:
        """Callback function for the timer."""
        timer_start = self.get_clock().now()
        self.publish_offboard_control_heartbeat_signal()
        
        if not self.is_vision_ready:
            # 只有在第一次进入timer_callback时执行
            if self.vision_controller.load_model():
                self.is_vision_ready = True
                self.get_logger().info("视觉系统准备就绪，开始执行任务逻辑。")
            else:
                self.get_logger().error("视觉系统初始化失败，节点将不执行任务。")
                return # 如果模型加载失败，直接返回，不执行后续逻辑          
        
        # 更新日志计数器
        self.log_counter += 1
        
        # --- 视觉处理部分 ---
        # ret, frame = self.cap.read()
        # if not ret:
        #     self.get_logger().warn("无法捕获图像")
        #     return

        

        # +++ (以下是新的替换代码) +++
        if self.latest_frame is None:
            self.get_logger().warn("尚未接收到任何图像帧...", throttle_duration_sec=2)
            return
        # (可选但推荐) 检查图像是否过时
        time_since_last_frame = (self.get_clock().now() - self.frame_received_time).nanoseconds / 1e9
        if time_since_last_frame > 1.0: # 如果超过1秒没有新图像
            self.get_logger().error("图像话题已超时！检查桥接或仿真是否正常。")
            return


        #进入offboard前发布位置控制点
             
        if self.offboard_setpoint_counter < 10:
            self.publish_position_setpoint(self.vehicle_local_position.x, self.vehicle_local_position.y, self.vehicle_local_position.z)
            self.engage_offboard_mode()  
            # 仅在日志计数满足条件时打印
            if self.log_counter % 10 == 0:
                self.get_logger().info(f"尝试切入offboard, ==============向前飞行距离{self.forward_x}m===================")

        if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:

            if self.current_dropping_state[1] != DroppingState.IDLE and not self.Is_Finish_1st_Drop:
                is_done = self.manage_dropping_sequence(1)
                if is_done:
                    self.get_logger().info("第一次投水流程确认完成。")
                    self.Is_Finish_1st_Drop = True


            if self.current_dropping_state[2] != DroppingState.IDLE and not self.Is_Finish_2nd_Drop:
                is_done = self.manage_dropping_sequence(2)
                if is_done:
                    self.get_logger().info("第二次投水流程确认完成。")
                    self.Is_Finish_2nd_Drop = True
                    # 更新任务完成标志
                    self.is_FinishDrop = True
            
            
            if not self.is_ReadyToTakeoff:
                if self.initial_x is None:
                    # 第一次进入此状态，记录当前位置为目标保持位置
                    self.initial_x = self.vehicle_local_position.x
                    self.initial_y = self.vehicle_local_position.y
                    self.initial_z = self.vehicle_local_position.z
                    self.init_yaw = self.vehicle_local_position.heading 
                    self.get_logger().info(f"进入Offboard模式，锁定初始位置: x={self.initial_x:.2f}, y={self.initial_y:.2f}, z={self.initial_z:.2f}")

                # 持续发布保持初始位置的指令
                self.publish_position_setpoint(self.initial_x, self.initial_y, self.initial_z)

                # 更新并检查位置稳定性
                current_pos = (
                    self.vehicle_local_position.x,
                    self.vehicle_local_position.y,
                    self.vehicle_local_position.z
                )
                self.initPositionChecker.update_position(current_pos)

                if self.initPositionChecker.is_stable():
                    self.is_ReadyToTakeoff = True
                    self.arm()
                    self.initial_x = self.vehicle_local_position.x
                    self.initial_y = self.vehicle_local_position.y
                    self.initial_z = self.vehicle_local_position.z
                    self.init_yaw = self.vehicle_local_position.heading
                    self.takeoff_target_height = float(self.initial_z + self.takeoff_height)
                    self.get_logger().info(f"起飞基准高度: {self.initial_z:.2f} m, 目标起飞高度: {self.takeoff_target_height:.2f} m")

            if self.is_ReadyToTakeoff and not self.is_AtTakeoffHeight:
                if self.log_counter % 25 == 0:
                    self.get_logger().info("执行步骤2,上升到指定高度")
                self.takeoff_relative()
                self.takeoff_height_check()
                # self.is_AtTakeoffHeight = False#  测试用

            if self.is_AtTakeoffHeight and not self.is_AtDropArea:
                if self.log_counter % 25 == 0:
                    self.get_logger().info("执行步骤3,飞向投水区")
                if not self.is_drop_area_calculated:
                    self.get_logger().info("执行步骤3, 计算投水区位置并开始导航...")
                    self.calculate_drop_area_once(self.forward_x)
                    self.is_drop_area_calculated = True

                # 步骤2: 持续导航并检查是否到达
                self.navigate_to_drop_area()
                # self.is_AtDropArea = False #测试用

            if self.is_AtDropArea and not self.is_FinishDrop:
                if self.mission_state == MissionState.START:
                    self.mission_state = MissionState.GLOBAL_SEARCH
                    self.get_logger().info(f"切换为GLOBAL_SEARCH模式。")
                    return
                
                #======启动投放区域计时模块========
                if self.drop_phase_start_time is None:
                    self.get_logger().info(f"已到达投水区域，启动 {self.drop_phase_timeout} 秒投放任务倒计时。")
                    self.drop_phase_start_time = self.get_clock().now()
                
                elapsed_drop_time = (self.get_clock().now() - self.drop_phase_start_time).nanoseconds / 1e9
                if elapsed_drop_time > self.drop_phase_timeout:
                    # self.get_logger().warn(f"投放阶段整体超时（超过 {self.drop_phase_timeout} 秒），进入强制投放流程。")
                    # <<< 修改：不再直接投放，而是切换到专用状态 >>>
                    self.mission_state = MissionState.TIMEOUT_DROP
                #======启动投放区域计时模块========
                
                ## 进入全局搜索模块
                if self.mission_state == MissionState.GLOBAL_SEARCH:
                    
                    # 开启usb摄像头识别
                    self.current_vision_info = self.latest_vision_info
                   
                    
                    #启动全局搜索计时器
                    if self.search_start_time is None:
                        self.get_logger().info(f"进入全局搜索，将持续 {self.search_timeout}s 建立稳定跟踪...")
                        self.search_start_time = self.get_clock().now()
                    
                    self.global_search_target_z = float(self.initial_z + self.global_search_height)
                    self.publish_position_setpoint(self.DropArea_x, self.DropArea_y, self.global_search_target_z)

                    elapsed_search_time = (self.get_clock().now() - self.search_start_time).nanoseconds / 1e9
                    
                    ## 全局搜索到达时间后
                    if elapsed_search_time > self.search_timeout:
                        self.get_logger().info("搜索时间到，开始根据跟踪历史和用户优先级构建最终任务地图。")
                        self._build_final_mission_map(self.current_vision_info)
                        
                        if not self.mission_targets_ned:
                            self.get_logger().error("搜索结束但未规划任何有效目标！进入超时投放。")
                            self.mission_state = MissionState.TIMEOUT_DROP
                        else:
                            # ==================== 启动平滑下降过程 ====================
                            self.get_logger().info("任务地图已构建，启动向首个目标的平滑移动。")
                        
                            first_target = self.mission_targets_ned[0]
                            target_x, target_y = first_target['coords_ned']
                            end_position = (target_x, target_y, self.takeoff_target_height)
                            
                            # 调用新的辅助函数来启动平滑移动
                            self._start_smooth_move(end_position)

                            self.mission_state = MissionState.TARGETING_CYCLE
                            # ========================================================
                        return
                
            
                elif self.mission_state == MissionState.TARGETING_CYCLE:
                    # ==================== 新增：平滑下降处理模块 ====================
                    if self.is_smoothing_descent:
                        # 计算当前进度 (从 0.0 到 1.0)
                        progress = self.smoothing_step_counter / self.smoothing_total_steps
                        progress = min(progress, 1.0) # 确保不会超过1.0

                        # 线性插值计算当前的中间目标点
                        start_x, start_y, start_z = self.smoothing_start_pos
                        end_x, end_y, end_z = self.smoothing_end_pos

                        interp_x = start_x * (1 - progress) + end_x * progress
                        interp_y = start_y * (1 - progress) + end_y * progress
                        interp_z = start_z * (1 - progress) + end_z * progress
                        
                        # 发布这个中间目标点
                        self.publish_position_setpoint(interp_x, interp_y, interp_z)

                        # 每隔一段时间打印日志，观察过程
                        if self.smoothing_step_counter % 25 == 0: # 大约每秒打印一次 (25 * 0.04s)
                            self.get_logger().info(f"平滑下降中 ({self.smoothing_step_counter}/{self.smoothing_total_steps})... "
                                                   f"目标高度: {interp_z:.2f} m")

                        # 更新步数
                        self.smoothing_step_counter += 1

                        # 检查平滑过程是否完成
                        if self.smoothing_step_counter > self.smoothing_total_steps:
                            self.publish_position_setpoint(end_x,end_y,end_z)
                            dist_err = math.hypot(self.vehicle_local_position.x - end_x, self.vehicle_local_position.y - end_y)
                            if dist_err < self.target_approach_threshold: # 到达阈值
                                self.get_logger().info(f"已到达目标上方，准备下降。")
                                self.is_navigating_to_target = False
                                self.is_final_aligning = True
                                self.is_smoothing_descent = False # 关闭平滑模式
                        # 在平滑下降期间，直接返回，不执行下面的对准逻辑
                        return 
                    # ================================================================
            
                    # 检查是否所有规划的目标都已打击，或已用完两次投放机会
                    if self.current_target_index >= len(self.mission_targets_ned) or self.visited_targets_count >= 2:
                        self.get_logger().info("所有已规划的目标均已打击，或已完成两次投放。任务完成。")
                        self.is_FinishDrop = True
                        return

                    # 获取当前要打击的目标
                    current_target = self.mission_targets_ned[self.current_target_index]
                    current_target_name = current_target['name']


                    if self.is_final_aligning:
                        # 3. 使用深度相机进行最终对准和投放
                        self.get_logger().info(f"正在对 '{current_target_name}' 进行最终对准...", throttle_duration_sec=2)
                        self.adjust_to_target() # 调用你已有的、基于/target_position的精确对准函数

                    
                    elif self.is_waiting_post_drop:
                        self.get_logger().info("投放时等待中...", throttle_duration_sec=1)
                        #判断是否完成第一/二次投放
                        is_first_drop_done = self.visited_targets_count == 0 and self.Is_Finish_1st_Drop
                        is_second_drop_done = self.visited_targets_count == 1 and self.Is_Finish_2nd_Drop
                        # 保持在当前位置悬停
                        self.publish_position_setpoint(
                            self.postdrop_waiting_x,
                            self.postdrop_waiting_y,
                            self.postdrop_waiting_z,
                        )
                        
                        # 检查延时是否结束
                        elapsed_delay = (self.get_clock().now() - self.post_drop_start_time).nanoseconds / 1e9
                        if elapsed_delay > self.post_drop_delay:
                            self.get_logger().info("停留结束。")
                            self.is_waiting_post_drop = False
                            if is_first_drop_done:
                                self.reset_for_next_target() # 现在才重置并开始下一个任务
                                if self.current_target_index < len(self.mission_targets_ned):
                                    self.get_logger().info("准备飞向下一个目标，再次启动平滑移动。")
                                    next_target = self.mission_targets_ned[self.current_target_index]
                                    target_x, target_y = next_target['coords_ned']
                                    end_position = (target_x, target_y, self.takeoff_target_height)
                                    # 再次调用新的辅助函数
                                    self._start_smooth_move(end_position)
                            elif is_second_drop_done:
                                self.get_logger().info(f"目标 '{current_target_name}' (第2个) 投放完成！")
                                # 此时不需要再 reset_for_next_target，直接标记总任务完成
                                self.is_FinishDrop = True
                                self.get_logger().info("所有预定目标均已打击。")




                elif self.mission_state == MissionState.TIMEOUT_DROP:
                    # self.get_logger().info("正在执行超时强制投放流程...")
                    if not self.Is_Finish_1st_Drop and self.current_dropping_state[1] == DroppingState.IDLE:
                        self.get_logger().warn("强制启动第一个载荷的投放序列。")
                        self.drop_payload(1)
                        self.timeout_drop_start_time = self.get_clock().now()

                    # 启动第二次强制投放 (如果第一个已完成且第二个还没开始)
                    if self.Is_Finish_1st_Drop and not self.Is_Finish_2nd_Drop and self.current_dropping_state[2] == DroppingState.IDLE:
                        if self.timeout_drop_start_time is None:
                            # 如果计时器未设置(说明超时发生在第一次投放完成后)，则立即设置它
                            self.get_logger().warn("超时流程启动时，第一次投放已完成。立即启动第二次投放延迟计时。")
                            self.timeout_drop_start_time = self.get_clock().now()
                        else:
                            elapsed_time = (self.get_clock().now() - self.timeout_drop_start_time).nanoseconds / 1e9
                            if elapsed_time > self.timeout_drop_delay:
                                self.get_logger().warn("强制启动第二个载荷的投放序列。")
                                self.drop_payload(2)
                            else:
                                return

                    # 3. 检查是否全部投放完毕
                    if self.Is_Finish_1st_Drop and self.Is_Finish_2nd_Drop:
                        self.get_logger().info("所有载荷均已强制投放，任务完成。")
                        self.is_FinishDrop = True # 触发外部状态机进入 DROP_COMPLETE

            if self.is_FinishDrop:
                if self.mission_state.value < MissionState.TRANSIT_TO_RECON_OFFBOARD.value:
                    self.get_logger().info("所有载荷投放完毕，准备在Offboard模式下飞往侦察区域...")

                    #回到投放区中心
                    self.publish_position_setpoint(self.DropArea_x,self.DropArea_y,self.takeoff_target_height)
                    error_2_center_DropArea = math.sqrt((self.vehicle_local_position.x-self.DropArea_x)**2+(self.vehicle_local_position.y-self.DropArea_y)**2+(self.vehicle_local_position.z-self.takeoff_target_height)**2)
                    if error_2_center_DropArea < 0.5 :
                        self.get_logger().info("已回到投放区中心")
                        self.mission_state = MissionState.RETURN_TO_CENTER_DROPAREA
                    else:
                        return
                        
                   
                if self.mission_state == MissionState.RETURN_TO_CENTER_DROPAREA:
                    # 1. 计算侦察区的目标点 (在当前位置的基础上向前飞)
                    # 注意：我们使用 coordinate_FRD2NED 函数，它会基于飞机的初始朝向 (init_yaw) 进行计算
                    # 首先获取飞机当前在初始FRD坐标系下的位置
                
                    # 计算目标FRD坐标
                    target_recon_x_frd = self.forward_x + self.recon_forward_distance # 向前飞
                    target_recon_y_frd = 0 # 侧向不变
                     # 将目标FRD坐标转换为全局NED坐标
                    target_recon_x_ned, target_recon_y_ned = self.coordinate_FRD2NED(
                        target_recon_x_frd,
                        target_recon_y_frd
                    )
                    
                    self.get_logger().info(f"将从当前位置向前飞 {self.recon_forward_distance}m, "
                                       f"目标侦察区 (NED): ({target_recon_x_ned:.2f}, {target_recon_y_ned:.2f})")
                    
                    self.publish_position_setpoint(target_recon_x_ned, target_recon_y_ned, self.takeoff_target_height)

                    error = math.sqrt((self.vehicle_local_position.x-target_recon_x_ned)**2+(self.vehicle_local_position.y-target_recon_y_ned)**2)
                    if error < 0.5 :
                    # 3. 切换到新的状态
                        self.mission_state = MissionState.TRANSIT_TO_RECON_OFFBOARD
                    else:
                        return

                # <<< MODIFIED: 只有需要在 OFFBOARD 模式下执行的侦察逻辑才留在这里 >>>
                # 状态：RECON_AREA_SWITCH_TO_OFFBOARD
                elif self.mission_state == MissionState.TRANSIT_TO_RECON_OFFBOARD:
                    self.get_logger().info("已成功切换回Offboard模式，开始侦察搜索。")
                    self.mission_state = MissionState.RECON_SEARCH

                # 状态：RECON_SEARCH
                elif self.mission_state == MissionState.RECON_SEARCH:
                    # ... (这部分逻辑不变) ...
                    if self.recon_search_start_time is None:
                        self.get_logger().info(f"爬升到侦察高度 {self.recon_search_height}m 并开始搜索...")
                        self.recon_search_start_time = self.get_clock().now()
                    self.publish_position_setpoint(self.vehicle_local_position.x, self.vehicle_local_position.y, self.initial_z + self.recon_search_height)
                    elapsed_search_time = (self.get_clock().now() - self.recon_search_start_time).nanoseconds / 1e9
                    self.current_vision_info = self.latest_vision_info
                    if elapsed_search_time > self.recon_search_timeout:
                        self.get_logger().info("侦察搜索时间到，构建侦察地图...")
                        self._build_recon_mission_map(self.current_vision_info)
                        if not self.recon_targets_ned:
                            self.get_logger().error("未发现任何侦察目标！任务结束。")
                            self.mission_state = MissionState.MISSION_COMPLETE
                        else:
                            self.get_logger().info("侦察地图构建完成，开始平滑飞越首个目标。")
                            
                            # Get the first target from the newly built list
                            first_target = self.recon_targets_ned[0]
                            target_name, (target_x, target_y) = first_target['name'], first_target['coords_ned']

                            # Use your helper function to start the smooth move
                            end_position = (target_x, target_y, self.takeoff_target_height)
                            self._start_smooth_move(end_position)
                            
                            self.mission_state = MissionState.RECON_CYCLE

                # 状态：RECON_CYCLE
                elif self.mission_state == MissionState.RECON_CYCLE:
                    
                    if self.current_recon_index >= len(self.recon_targets_ned):
                        self.get_logger().info("所有侦察目标均已访问，任务完成！")
                        self.mission_state = MissionState.MISSION_COMPLETE
                        return 
                    
                    if self.is_smoothing_descent:
                        # 计算当前进度 (从 0.0 到 1.0)
                        progress = self.smoothing_step_counter / self.smoothing_total_steps
                        progress = min(progress, 1.0)

                        # 线性插值计算当前的中间目标点
                        start_x, start_y, start_z = self.smoothing_start_pos
                        end_x, end_y, end_z = self.smoothing_end_pos
                        interp_x = start_x * (1 - progress) + end_x * progress
                        interp_y = start_y * (1 - progress) + end_y * progress
                        interp_z = start_z * (1 - progress) + end_z * progress
                        
                        self.publish_position_setpoint(interp_x, interp_y, interp_z)
                        self.smoothing_step_counter += 1

                        # 当平滑移动时间结束时，关闭标志位。
                        # 后续逻辑将负责确认最终到达。
                        if self.smoothing_step_counter > self.smoothing_total_steps:
                            self.get_logger().info("平滑移动阶段完成，现在确认最终到达。")
                            self.is_smoothing_descent = False
                        
                        return # 在平滑移动期间，跳过后续逻辑

                    # 获取当前目标信息
                    target = self.recon_targets_ned[self.current_recon_index]
                    target_name, (target_x, target_y) = target['name'], target['coords_ned']

                    if not self.is_hovering_at_recon_point:
                        # STATE: MOVING & ARRIVING
                        # 平滑移动已结束，现在我们发布最终目标点并等待无人机精确到达。
                        self.get_logger().info(f"正在接近侦察目标 {self.current_recon_index + 1}/{len(self.recon_targets_ned)}: '{target_name}'", throttle_duration_sec=2)
                        self.publish_position_setpoint(target_x, target_y, self.takeoff_target_height)
                        
                        dist_err = math.hypot(self.vehicle_local_position.x - target_x, self.vehicle_local_position.y - target_y)
                        if dist_err < self.recon_nav_threshold:
                            # 已到达！切换到悬停状态。
                            self.get_logger().info(f"已到达 '{target_name}' 上方，开始悬停侦察 {self.recon_hover_time} 秒。")
                            self.is_hovering_at_recon_point = True
                            self.recon_hover_start_time = self.get_clock().now()
                    else:
                        # STATE: HOVERING
                        elapsed_hover_time = (self.get_clock().now() - self.recon_hover_start_time).nanoseconds / 1e9
                        if elapsed_hover_time < self.recon_hover_time:
                            # 保持悬停
                            self.get_logger().info(f"正在侦察 '{target_name}'... {elapsed_hover_time:.1f}s", throttle_duration_sec=1)
                            self.publish_position_setpoint(target_x, target_y, self.takeoff_target_height)
                        else:
                            # 悬停结束，准备飞往下一个目标
                            self.get_logger().info(f"'{target_name}' 侦察完毕。")
                            self.current_recon_index += 1
                            self.is_hovering_at_recon_point = False # 切换回“移动”状态
                            
                            # 如果还有下一个目标，则为它启动平滑移动
                            if self.current_recon_index < len(self.recon_targets_ned):
                                next_target = self.recon_targets_ned[self.current_recon_index]
                                next_target_name, (next_target_x, next_target_y) = next_target['name'], next_target['coords_ned']
                                self.get_logger().info(f"准备平滑移动至下一个目标: '{next_target_name}'")
                                end_position = (next_target_x, next_target_y, self.takeoff_target_height)
                                self._start_smooth_move(end_position)

                # 状态：MISSION_COMPLETE
                elif self.mission_state == MissionState.MISSION_COMPLETE:
                    
                    
                    if not self.return_to_recon_center:
                        target_recon_x_frd = self.forward_x + self.recon_forward_distance # 向前飞
                        target_recon_y_frd = 0 # 侧向不变
                        # 将目标FRD坐标转换为全局NED坐标
                        target_recon_x_ned, target_recon_y_ned = self.coordinate_FRD2NED(
                            target_recon_x_frd,
                            target_recon_y_frd
                        )
                        self.fly_to_position(target_recon_x_ned,target_recon_y_ned,self.takeoff_target_height)
                        dist_err = math.hypot(self.vehicle_local_position.x - target_recon_x_ned, self.vehicle_local_position.y - target_recon_y_ned)
                        if dist_err < self.recon_nav_threshold:
                            self.return_to_recon_center = True
                            self.get_logger().info("已经回到侦察区域中心")
                    else:
                        self.return_to_launch()
                        self.get_logger().info("所有任务阶段均已完成，RTL。")
        
        else:
            # 只有在过了初始的切换阶段后才打印日志，避免启动时的干扰
            if self.offboard_setpoint_counter >= 10:
                self.publish_position_setpoint(
                    self.vehicle_local_position.x,
                    self.vehicle_local_position.y,
                    self.vehicle_local_position.z
                )
                retry_interval = max(1, int(1.0 / self.dt))
                if self.offboard_setpoint_counter % retry_interval == 0:
                    self.engage_offboard_mode()
                self.get_logger().warn(f"无人机当前状态 ({self.vehicle_status.nav_state}) 不是预期的 Offboard 或航线模式。", throttle_duration_sec=5)
            
        
        self.offboard_setpoint_counter += 1
        
        # =================== 显示图像 ===================
    # 显示由视觉定时器生成的最新标注图像
        if self.show_video:
            if self.latest_annotated_frame is not None:
                cv2.imshow("Drone View", self.latest_annotated_frame)
                cv2.waitKey(1)
        elasped_timer_time = (self.get_clock().now() - timer_start).nanoseconds / 1e9
        if self.offboard_setpoint_counter % 150 == 0:
            self.get_logger().info(f"控制循环花费时间：{elasped_timer_time:.5f}s")
        
    def vision_timer_callback(self):
        """
        这个回调以较低频率运行，专门处理耗时的视觉任务。
        """
        timer_start = self.get_clock().now()
        if not self.is_vision_ready or self.latest_frame is None:
            return

        # 复制帧以进行处理
        frame_to_process = self.latest_frame.copy()
        
        # <<< 核心决策逻辑 >>>
        num_targets_for_vision = 0 # 默认不处理
        
        # 状态1：投水前的全局搜索，需要找 3 个目标
        if self.mission_state == MissionState.GLOBAL_SEARCH:
            num_targets_for_vision = 3
        
        # 状态2：侦察阶段的搜索，需要找 5 个目标
        elif self.mission_state == MissionState.RECON_SEARCH:
            num_targets_for_vision = 5

        # 核心视觉处理调用
        current_altitude = self.vehicle_local_position.z - (self.initial_z or 0)
        vision_info, annotated_frame = self.vision_controller.process_frame(
            frame_to_process, 
            current_altitude,
            max_targets_to_confirm=num_targets_for_vision # <<< 将决策结果传入
        )

        # 只有在进行有效处理时才更新视觉信息
        if num_targets_for_vision > 0:
            self.latest_vision_info = vision_info
        
        # 更新用于显示的 annotated_frame (无论是否处理都更新，以便显示状态)
        cv2.putText(annotated_frame, f"State: {self.mission_state.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Vision Targets: {num_targets_for_vision}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2) # 添加一个状态显示
        self.latest_annotated_frame = annotated_frame

        elasped_timer_time = (self.get_clock().now() - timer_start).nanoseconds / 1e9
        if self.offboard_setpoint_counter % 150 == 0:
            self.get_logger().info(f"视觉循环花费时间：{elasped_timer_time:.5f}")


def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)

    # 2. 设置我们自己的命令行参数解析器
    parser = argparse.ArgumentParser(description="Offboard control script for PX4 drone mission.")

    pkg_name = 'control'  # 替换的包名
        
    try:
        pkg_share_dir = get_package_share_directory(pkg_name)
        
        # 假设你的pt文件在包的根目录
        default_weights_path = os.path.join(pkg_share_dir, 'models', 'best_sim.pt')
        
        # 检查文件是否存在，不存在则使用备用路径
        if not os.path.exists(default_weights_path):
            self.get_logger().warn(f"Default weights file not found at {default_weights_path}")
            # 可以设置为空字符串或其他默认值
            default_weights_path = ""
            
    except PackageNotFoundError:
        self.get_logger().error(f"Package {pkg_name} not found")
        default_weights_path = ""

    
    # 添加你想通过命令行配置的参数
    parser.add_argument('--model-path', type=str, default=default_weights_path,
                        help='Path to the object detection model file.')
    parser.add_argument('--photo-path', type=str, default='~/image_recodes',
                        help='Base directory to save captured photos.')
    parser.add_argument('--video-path', type=str, default='~/video_recodes',
                        help='Base directory to save recorded mission videos.')
    parser.add_argument('--camera-hint', type=str, default='imx577',
                        help='Hint to find the camera device name (e.g., "USB", "C920").')
    
    parser.add_argument('--takeoff-height', type=float, default=-2.8,
                        help='Takeoff height in meters (negative value for altitude).')
    parser.add_argument('--descent-height', type=float, default=0.8,
                        help='Descent height after first alignment in meters (positive value).')
    
    parser.add_argument('--forward-x', type=float, default=2.5,
                        help='Forward distance to fly to the drop area in meters.')
    parser.add_argument('--search-height', type=float, default=-4.0,
                        help='Global search height in meters (negative value for altitude).')
   
    parser.add_argument('--align-maxstep', type=float, default=0.2,
                        help='Maximum step size for each alignment adjustment.')

    
    # <<< 新增：在这里为 AlignmentChecker 添加参数 >>>
    parser.add_argument('--first-align-threshold', type=float, default=0.15,
                        help='Threshold (distance in meters) for the first alignment.')
    parser.add_argument('--first-align-time-window', type=float, default=2.0,
                        help='Time window (seconds) to maintain stability for the first alignment.')
    parser.add_argument('--first-align-check-freq', type=int, default=5,
                        help='Check frequency (how many timer calls per check) for the first alignment.')
    parser.add_argument('--second-align-threshold', type=float, default=0.10,
                        help='Threshold (distance in meters) for the second alignment.')
    parser.add_argument('--second-align-time-window', type=float, default=3.0,
                        help='Time window (seconds) to maintain stability for the second alignment.')
    parser.add_argument('--second-align-check-freq', type=int, default=5,
                        help='Check frequency (how many timer calls per check) for the second alignment.')    
    
    parser.add_argument('--drop-phase-timeout', type=float, default=80,
                        help='Maximum time in seconds for the entire dropping phase.')
    parser.add_argument('--search-timeout', type=float, default=5.0,
                        help='Maximum time in seconds for each search attempt.')
    parser.add_argument('--second-align-maxtime', type=float, default=15,
                        help='Maximum time in seconds for each search attempt.')
    parser.add_argument('--first-align-maxtime', type=float, default=5, 
                        help='Maximum time in seconds for the first alignment phase before forcing a drop.')
    
    
    parser.add_argument('--depthcam_xoffset', type=float, default=-0.065,
                        help='深度相机的x方向误差.')
    parser.add_argument('--depthcam_yoffset', type=float, default=0.033,
                        help='深度相机的y方向误差.')
    
    
     # --- 定时器参数 ---
    parser.add_argument('--timer-period', type=float, default=0.03,
                        help='定时器周期 (秒), 这也决定了PID控制中的 dt。默认: 0.03s (约33Hz).')
    parser.add_argument('--vision-timer-period', type=float, default=0.1,
                        help='定时器周期 (秒), 默认: 0.1s (10Hz).')


    # --- PID 核心参数 ---
    parser.add_argument('--kp', type=float, default=0.5,
                        help='PID控制器 - 精细调节阶段的P增益 (Kp)。默认: 0.9911.')
    parser.add_argument('--ki', type=float, default=0.0,
                        help='PID控制器 - 积分增益 (Ki)。默认: 0.1021.')
    parser.add_argument('--kd', type=float, default=0.0000,
                        help='PID控制器 - 微分增益 (Kd)。默认: 0.0009.')
    parser.add_argument('--kf', type=float, default=0.0,
                    help='前馈控制器 - 基于速度的阻尼增益 (Kf)。建议范围: 0.1 - 0.5')

    # --- PID 行为阈值和限制参数 ---
    parser.add_argument('--max-integral', type=float, default=0.2, # 这个值默认等于 align_maxstep
                        help='PID控制器 - 积分项的最大限制值 (防止积分饱和)。默认: 0.2.')
    
    parser.add_argument('--tracking-buffer', type=int, default=30, help='Number of frames for tracking history.')

    parser.add_argument('--post-drop-delay', type=float, default=1.0,
                        help='每次投放后悬停等待的时间（秒）。')
    parser.add_argument('--timeout-drop-delay', type=float, default=1.0,
                        help='在超时强制投放流程中，两次投放之间的最小间隔（秒）。')
    parser.add_argument('--servo-step-delay', type=float, default=0.1,
                        help='舵机每个动作之间的延迟时间（秒）。')
    
    parser.add_argument('--takeoff-threshold', type=float, default=0.22,
                        help='判断无人机到达起飞高度的误差阈值（米）。')
    parser.add_argument('--nav-threshold', type=float, default=0.2,
                        help='判断无人机到达导航点（如投水区）的误差阈值（米）。')
    parser.add_argument('--target-approach-threshold', type=float, default=0.3,
                        help='判断无人机飞到目标上方，可以开始精确对准的误差阈值（米）。')
    parser.add_argument('--alignment-altitude-threshold', type=float, default=0.5,
                        help='在检查X/Y对准前，无人机必须达到的高度误差阈值（米）。')
    parser.add_argument('--headless', action='store_true',
                        help='以无头模式运行，不显示摄像头的GUI窗口。')
    
    # <<< 新增：用于控制视频录制的参数 >>>
    parser.add_argument('--record-video', action='store_true',
                        help='启用任务视频录制功能。')
    
    # --- 选择投放桶 --- 
    parser.add_argument('--target-order', 
                        type=int,  # 关键：将类型改为整数
                        nargs='+', # 接收一个或多个值
                        default=[1, 3, 2], # 默认顺序: 中(2), 左(1), 右(3)
                        help='设置目标的投放顺序。使用数字: 1=左, 2=中, 3=右。 '
                             '例如: --target-order 3 1 2')
    
    # === 新增：为侦察任务添加参数 ===
    parser.add_argument('--recon-search-height', type=float, default=-5.0,
                        help='执行第二次（侦察）视觉搜索时的高度（米）。')
    parser.add_argument('--recon-search-timeout', type=float, default=5.0,
                        help='侦察阶段视觉搜索的持续时间（秒）。')
    parser.add_argument('--recon-hover-time', type=float, default=3.0,
                        help='到达每个侦察圆筒上方后的悬停侦察时间（秒）。')
    parser.add_argument('--recon-nav-threshold', type=float, default=0.5,
                        help='判断无人机到达侦察点的误差阈值（米）。')
    
    parser.add_argument('--recon-forward-distance', type=float, default=6.0,
                        help='投水完成后，在Offboard模式下向前飞行以到达侦察区的距离（米）。')
    
    
    # <<< 新增：动态平滑移动的参数 >>>
    parser.add_argument('--smoothing-speed', type=float, default=1.5,
                        help='Average speed (m/s) for smooth transitions between targets.')
    parser.add_argument('--min-smoothing-duration', type=float, default=1.0,
                        help='Minimum duration (seconds) for any smooth move to ensure stability.')
    parser.add_argument('--max-smoothing-duration', type=float, default=8.0,
                        help='Maximum duration (seconds) for any smooth move to cap long-distance travel time.')
    
    # 3. 解析参数
    # 使用 rclpy.utilities.remove_ros_args 来确保我们只解析自己的参数，
    # 这样可以安全地与 ROS2 的参数（如 --ros-args）一起使用。
    custom_args = parser.parse_args(args=rclpy.utilities.remove_ros_args(args=sys.argv)[1:])

    TARGET_MAP = {
        1: "Left",
        2: "Middle",
        3: "Right"
    }
    VALID_INPUTS = set(TARGET_MAP.keys()) # {1, 2, 3}

    user_order_nums = custom_args.target_order

    # 验证1：检查用户输入的数字是否都在允许的范围内
    for num in user_order_nums:
        if num not in VALID_INPUTS:
            print(f"错误：无效的顺序编号 '{num}'。请从 {list(VALID_INPUTS)} 中选择。")
            sys.exit(1) # 退出程序

    # 验证2：确保没有重复的编号，并且数量正确 (正好是3个)
    if len(set(user_order_nums)) != len(VALID_INPUTS):
        print(f"错误：投放顺序必须包含且仅包含 {list(VALID_INPUTS)} 各一次。")
        print(f"您提供的顺序是: {user_order_nums}")
        sys.exit(1) # 退出程序

    # 翻译：将数字列表 [3, 1, 2] 转换为字符串列表 ["Right", "Left", "Middle"]
    try:
        translated_order_strings = [TARGET_MAP[num] for num in user_order_nums]
    except KeyError as e:
        # 这一步理论上不会出错，因为上面已经验证过了，但作为健壮性代码保留
        print(f"内部错误：无法翻译编号 {e}。")
        sys.exit(1)

    # 关键：用翻译好的字符串列表，覆盖掉原来的数字列表
    custom_args.target_order = translated_order_strings
    
    # =================================================================
    # ##########################################################################
    # ################          新增的任务参数总览打印模块          ################
    # ##########################################################################
    print("\n================== 任务参数总览 ==================")
    print(f"  - 模型文件: {custom_args.model_path}")
    print(f"  - 目标投放顺序: {custom_args.target_order}")
    print("------------------ 飞行参数 ------------------")
    print(f"  - 计划向前飞行距离: {custom_args.forward_x} 米")
    print(f"  - 预设起飞高度: {abs(custom_args.takeoff_height)} 米 (相对于初始位置)")
    print(f"  - 全局搜索高度: {abs(custom_args.search_height)} 米 (相对于初始位置)")
    print(f"  - 首次对准后下降: {custom_args.descent_height} 米")
    print("------------------ 超时设置 ------------------")
    print(f"  - 整体投放阶段超时: {custom_args.drop_phase_timeout} 秒")
    print(f"  - 全局搜索阶段超时: {custom_args.search_timeout} 秒")
    print(f"  - 首次对准阶段超时: {custom_args.first_align_maxtime} 秒")
    print(f"  - 第二次对准阶段超时: {custom_args.second_align_maxtime} 秒")
    print("------------------ 对准阈值 ------------------")
    print(f"  - 首次对准稳定阈值: {custom_args.first_align_threshold} 米, 稳定时长: {custom_args.first_align_time_window} 秒")
    print(f"  - 第二次对准稳定阈值: {custom_args.second_align_threshold} 米, 稳定时长: {custom_args.second_align_time_window} 秒")
    print("------------------ 模式设置 ------------------")
    print(f"  - 视频录制: {'已启用' if custom_args.record_video else '已禁用'}")
    print(f"  - 无头模式 (不显示GUI): {'是' if custom_args.headless else '否'}")
    print("==================================================\n")
    # ##########################################################################
    
    print('Starting offboard control node with custom parameters...')
    
    # 4. 将解析后的参数传入节点
    offboard_control = OffboardControl(args=custom_args)

    try:
        rclpy.spin(offboard_control)
    except KeyboardInterrupt:
        print("程序被用户中断 (Ctrl+C)")
    finally:
        # 确保节点在退出时被正确销毁，从而触发我们的清理逻辑
        offboard_control.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
