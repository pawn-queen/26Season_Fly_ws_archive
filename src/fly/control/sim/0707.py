#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus, VehicleOdometry, VehicleLandDetected
from geometry_msgs.msg import Point
from std_msgs.msg import Float32, String
from collections import deque
import time
from control.DronePositionChecker import DronePositionChecker
from control.AlignmentChecker import AlignmentChecker
from control.ServoControl import ServoControl
from control.gui_support import opencv_gui_available
from control.target_anchor import (
    TargetAnchorTracker,
    altitude_within_threshold,
    px4_pose_attitude_timestamps_match,
)
from control.sim.drop_evaluator import KinematicDropEvaluator #从包内导入drop_evaluator判断落点
from control.visual_servoing import VisualServoingController # 从你的包中导入视觉控制器
import cv2
from enum import Enum
import subprocess
import re
import os
import csv
import argparse # <<< 新增
import sys      # <<< 新增
import traceback
import numpy as np
import threading
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, PointCloud
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, ShutdownException, MultiThreadedExecutor
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory


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
    REQUEST_RTL = 15
    RTL_ACTIVE = 16
    DONE = 17

MISSION_STATE_DESCRIPTIONS = {
    MissionState.START: "等待进入任务",
    MissionState.TAKING_OFF: "起飞阶段",
    MissionState.GLOBAL_SEARCH: "投放目标全局搜索",
    MissionState.TARGETING_CYCLE: "投放目标导航/对准",
    MissionState.TIMEOUT_DROP: "投放超时强制投放",
    MissionState.TRANSIT_TO_RECON_OFFBOARD: "准备进入侦察搜索",
    MissionState.RETURN_TO_CENTER_DROPAREA: "返回投放区中心",
    MissionState.RECON_SEARCH: "侦察搜索/建图",
    MissionState.RECON_CYCLE: "侦察目标巡航",
    MissionState.MISSION_COMPLETE: "任务阶段完成",
    MissionState.REQUEST_RTL: "请求返航",
    MissionState.RTL_ACTIVE: "返航中",
    MissionState.DONE: "任务结束",
}

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
        target_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # 飞行状态机、YOLO 推理、图像接收和 Offboard 心跳必须互相隔离。
        # 尤其不能让耗时推理或 OpenCV GUI 阻塞位置控制状态机。
        self.mission_callback_group = MutuallyExclusiveCallbackGroup()
        self.vision_callback_group = MutuallyExclusiveCallbackGroup()
        self.heartbeat_callback_group = MutuallyExclusiveCallbackGroup()
        self.image_callback_group = MutuallyExclusiveCallbackGroup()
        self._frame_lock = threading.Lock()
        self._vision_result_lock = threading.Lock()
        self._map_data_lock = threading.Lock()
        self.offboard_heartbeat_enabled = False

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.mission_state_publisher = self.create_publisher(String, '/mission_state', 10)
        self.simulated_drop_result_publisher = self.create_publisher(
            String, '/simulated_drop_result', 10)

        # Create subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self.vehicle_local_position_callback,
            qos_profile, callback_group=self.mission_callback_group)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v1', self.vehicle_status_callback,
            qos_profile, callback_group=self.mission_callback_group)
        self.target_position_subscriber = self.create_subscription(Point, '/target_position',
                                                                   self.target_position_callback,
                                                                   target_qos_profile,
                                                                   callback_group=self.mission_callback_group)
        self.target_observation_subscriber = self.create_subscription(
            PointCloud,
            '/target_observation',
            self.target_observation_callback,
            target_qos_profile,
            callback_group=self.mission_callback_group,
        )
        
        self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.vehicle_odometry_callback,
            qos_profile, callback_group=self.mission_callback_group)
        self.vehicle_land_detected_subscriber = self.create_subscription(
            VehicleLandDetected, '/fmu/out/vehicle_land_detected', self.vehicle_land_detected_callback,
            qos_profile, callback_group=self.mission_callback_group)
        
        #这是广角相机的内参和畸变参数
        self.camera_matrix = np.array([
            [465.7411193847656, 0., 320.0],
            [0., 465.7411193847656, 240.0],
            [0., 0., 1.]
        ])
        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) # 假设畸变可以忽略

        # 广角相机外参：机体坐标系采用 FRD（前、右、下）。安装角用于
        # 补偿仿真传感器与默认向下安装姿态之间的差异。
        widecam_nominal_rotation = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        widecam_mount_correction = R.from_euler(
            'xyz',
            [args.widecam_roll_deg, args.widecam_pitch_deg, args.widecam_yaw_deg],
            degrees=True,
        ).as_matrix()
        self.R_body_widecam = widecam_mount_correction @ widecam_nominal_rotation
        self.p_widecam_in_body = np.array([
            args.widecam_xoffset,
            args.widecam_yoffset,
            args.widecam_zoffset,
        ])

        self.STATIC_OFFSET_X_FRD = args.depthcam_xoffset # 假设这是旧的x_offset (对应机体前方)
        self.STATIC_OFFSET_Y_FRD = args.depthcam_yoffset  # 假设这是旧的y_offset (对应机体右方)
        
        CAM_POS_IN_BODY = np.array([
            self.STATIC_OFFSET_X_FRD,
            self.STATIC_OFFSET_Y_FRD,
            args.depthcam_zoffset,
        ])   # 相机位置 (前, 右, 下) in meters
        DROPPER_POS_IN_BODY = np.array([
            args.dropper_xoffset,
            args.dropper_yoffset,
            args.dropper_zoffset,
        ]) # 虚拟投放器位置 (前, 右, 下) in meters

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
        self.vehicle_attitude_timestamp_us = None
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
            tracking_buffer_size=args.tracking_buffer,
            camera_to_body_rotation=self.R_body_widecam,
            camera_position_in_body=self.p_widecam_in_body,
            min_ground_ray_down=args.widecam_min_ground_ray_down,
            max_ground_range_m=args.widecam_max_ground_range,
        )
        self.widecam_map_max_speed = args.widecam_map_max_speed
        self.widecam_map_outlier_floor_m = args.widecam_map_outlier_floor
        self.widecam_pose_attitude_skew_us = int(args.widecam_max_pose_attitude_skew * 1e6)
        self.get_logger().info(
            "Wide-camera map filters: min_down=%.3f, max_range=%.1fm, max_speed=%.2fm/s, "
            "outlier_floor=%.2fm, pose_attitude_skew=%.3fs" % (
                args.widecam_min_ground_ray_down,
                args.widecam_max_ground_range,
                self.widecam_map_max_speed,
                self.widecam_map_outlier_floor_m,
                args.widecam_max_pose_attitude_skew,
            )
        )
        
        # device_path = self.find_video_device_by_name(args.camera_hint)
        # self.cap = cv2.VideoCapture(device_path if device_path else 0)        
        # if not self.cap.isOpened():
        #     self.get_logger().error("无法打开摄像头！")
        #     rclpy.shutdown()

        self.bridge = CvBridge()
        self.latest_frame = None  # 用于存储最新接收到的图像帧
        self.frame_received_time = self.get_clock().now() # 用于检查图像是否过时
        # 每张图像必须与拍摄时的飞行器位姿关联；不能使用推理完成时的最新位姿。
        self.latest_frame_pose_snapshot = None
        self.latest_frame_sequence = 0
        self.last_processed_frame_sequence = 0
        self.image_timeout_sec = max(0.5, float(args.image_timeout))
        
        # 创建图像话题订阅者
        self.image_subscriber = self.create_subscription(
            Image,
            sim_camera_topic, # 订阅来自仿真的图像话题
            self.image_callback,
            qos_profile_sensor_data,  # 使用 sensor_data QoS 配置
            callback_group=self.image_callback_group,
        )
        self.get_logger().info(f"订阅仿真摄像头话题: '{sim_camera_topic}'")

        
        self.is_vision_ready = False

        # === 新增：任务流程管理变量 ===
        self.mission_state = MissionState.START
        self.mission_state_timer = self.create_timer(
            0.2, self.publish_mission_state, callback_group=self.mission_callback_group)
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
        self.vehicle_land_detected = VehicleLandDetected()
        
        self.target_position = None
        self.target_confidence = None
        self.target_observation_sequence = 0
        self.last_logged_target_observation_sequence = -1
        self.last_confident_target_update_time = None
        self.last_target_measurement_time_s = None
        self.final_alignment_started_at_s = None
        self.target_pose_history = deque(maxlen=400)
        self.target_pose_max_skew = args.target_pose_max_skew
        self.target_pose_attitude_max_skew = (
            args.target_pose_attitude_max_skew
        )
        self.target_observation_frame_id = args.target_observation_frame_id.strip()
        self.target_stream_discontinuity_pending = False
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
        self.first_align_accumulated_s = 0.0
        self.second_align_accumulated_s = 0.0
        
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
        self.rtl_request_sent = False
        self.rtl_active_logged = False
        self.done_logged = False

        self.Is_Finish_1st_Drop = False
        self.Is_Finish_2nd_Drop = False

        self.search_start_time = None

        self.last_target_update_time = None
        self.target_timeout_duration = args.target_timeout_duration
        self.target_anchor_tracker = TargetAnchorTracker(
            confidence_window_s=args.target_confidence_window,
            hold_duration_s=args.target_anchor_hold_duration,
        )
        self.target_anchor_jump_pending = False
        self.target_anchor_reset_state = None
        self.alignment_target_was_fresh = False
        
        
        ### 新增: 用于稳定建图的数据收集变量 ###
        self.map_data_collection = []  # 存储多帧的坐标地图
        # 全局搜索阶段中，按世界 NED 坐标保存每个目标的多帧平均结果。
        self.world_target_coordinates_ned = {}
        self.widecam_map_reset_state = None
        self.widecam_local_reference_warned = False

        #==================投水状态机=================
        self.servo_step_delay = args.servo_step_delay  # 每个舵机动作之间的延迟（秒），可以根据实际情况调整
        self.current_dropping_state = {1: DroppingState.IDLE, 2: DroppingState.IDLE}
        self.last_servo_command_time = {1: None, 2: None}

        # ========== 目标像素坐标日志 ==========
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.expanduser('~/flylogs')
        log_filename = f'bucket_pixel_log_{timestamp}.csv'
        os.makedirs(log_dir, exist_ok=True)
        self.pixel_log_path = os.path.join(log_dir, log_filename)

        # 打开文件并保留文件句柄和writer对象
        self.pixel_log_file = open(self.pixel_log_path, 'w', newline='', encoding='utf-8')
        self.pixel_log_writer = csv.writer(self.pixel_log_file)
        # 写入表头
        self.pixel_log_writer.writerow([
            'timestamp',
            'target_x',
            'target_y',
            'target_confidence',
            'alignment_stage',
        ])
        self.get_logger().info(f"日志文件已创建并打开: {self.pixel_log_path}")
        # ===========================================================================

        # x500_depth 世界中没有可释放的水体和碰撞回传。每次实际发送投放
        # 指令时，使用释放瞬间的 PX4 状态预测虚拟载荷落点并把可复核数据写入 CSV。
        self.drop_eval_enabled = args.simulated_drop_evaluation
        self.drop_evaluator = KinematicDropEvaluator(
            hit_radius_m=args.sim_drop_hit_radius,
            gravity_mps2=args.sim_drop_gravity,
            wind_north_mps=args.sim_drop_wind_north,
            wind_east_mps=args.sim_drop_wind_east,
        )
        self.sim_drop_ground_z = args.sim_drop_ground_z
        self.sim_drop_target_max_age = args.sim_drop_target_max_age
        self.drop_evaluations = {}
        self.latest_drop_evaluation = None
        drop_eval_dir = os.path.expanduser(args.drop_eval_log_dir)
        os.makedirs(drop_eval_dir, exist_ok=True)
        self.drop_eval_path = os.path.join(drop_eval_dir, f'simulated_drop_eval_{timestamp}.csv')
        self.drop_eval_file = open(self.drop_eval_path, 'w', newline='', encoding='utf-8')
        self.drop_eval_writer = csv.DictWriter(
            self.drop_eval_file,
            fieldnames=[
                'timestamp_sec', 'drop_number', 'reason', 'target_name', 'target_age_s',
                'alignment_complete', 'ground_plane_source', 'ground_down_m',
                'status', 'hit', 'flight_time_s',
                'release_north_m', 'release_east_m', 'release_down_m',
                'impact_north_m', 'impact_east_m',
                'target_north_m', 'target_east_m', 'radial_error_m', 'hit_radius_m', 'message',
            ],
        )
        self.drop_eval_writer.writeheader()
        self.drop_eval_file.flush()
        self.get_logger().info(
            f"虚拟投放评估{'已启用' if self.drop_eval_enabled else '已禁用'}，结果文件: {self.drop_eval_path}"
        )

        # Create a timer to publish control commands
        self.dt = args.timer_period             # 控制周期 (秒) - 与timer频率一致
        self.control_timer = self.create_timer(
            self.dt, self.control_timer_callback, callback_group=self.mission_callback_group)
        self.offboard_heartbeat_timer = self.create_timer(
            min(self.dt, 0.05), self.offboard_heartbeat_timer_callback,
            callback_group=self.heartbeat_callback_group)
        
        # 创建一个新的、较慢的视觉处理定时器
        self.vision_processing_period = args.vision_timer_period # 10Hz, 可根据设备性能调整
        self.vision_timer = self.create_timer(
            self.vision_processing_period, self.vision_timer_callback,
            callback_group=self.vision_callback_group)
        
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
            check_frequency=args.first_align_check_freq,
            time_func=lambda: self.get_clock().now().nanoseconds / 1e9,
        )
        self.second_alignment_checker = AlignmentChecker(
            logger_func=self.get_logger().info,
            threshold=args.second_align_threshold,
            time_window=args.second_align_time_window,
            check_frequency=args.second_align_check_freq,
            time_func=lambda: self.get_clock().now().nanoseconds / 1e9,
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
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame_pose = self._capture_widecam_pose_snapshot()
            with self._frame_lock:
                self.latest_frame = frame
                self.frame_received_time = self.get_clock().now()
                self.latest_frame_pose_snapshot = frame_pose
                self.latest_frame_sequence += 1
        except Exception as e:
            self.get_logger().error(f"无法转换图像: {e}")
            



    def _record_target_pose_sample(self):
        """Store the current vehicle pose on the ROS clock for image matching."""
        position = self.vehicle_local_position
        position_timestamp_us = (
            getattr(position, 'timestamp_sample', 0)
            or getattr(position, 'timestamp', 0)
        )
        if not px4_pose_attitude_timestamps_match(
            position_timestamp_us,
            self.vehicle_attitude_timestamp_us,
            self.target_pose_attitude_max_skew,
        ):
            return

        body_to_ned = self._body_to_ned_rotation()
        position_ned = np.array([position.x, position.y, position.z], dtype=float)
        position_valid = (
            getattr(position, 'xy_valid', True)
            and getattr(position, 'z_valid', True)
        )
        if (
            body_to_ned is None
            or not position_valid
            or not np.all(np.isfinite(position_ned))
        ):
            return
        now_s = self.get_clock().now().nanoseconds / 1e9
        reset_state = self._local_position_reset_state()
        if self.target_pose_history:
            last_time_s, _, _, last_reset_state = self.target_pose_history[-1]
            if now_s < last_time_s or reset_state != last_reset_state:
                self.target_pose_history.clear()
        self.target_pose_history.append(
            (now_s, position_ned, body_to_ned, reset_state)
        )

    def _target_pose_at(self, measurement_time_s):
        """Return the pose nearest an image timestamp, or ``None`` if too far."""
        if measurement_time_s is None:
            if not self.target_pose_history:
                return None
            sample_time_s, position_ned, body_to_ned, reset_state = (
                self.target_pose_history[-1]
            )
            sample_age_s = (
                self.get_clock().now().nanoseconds / 1e9 - sample_time_s
            )
            if not 0.0 <= sample_age_s <= self.target_pose_max_skew:
                return None
            return position_ned, body_to_ned, reset_state

        if not math.isfinite(measurement_time_s):
            return None
        if not self.target_pose_history:
            return None
        sample_time_s, position_ned, body_to_ned, reset_state = min(
            self.target_pose_history,
            key=lambda sample: abs(sample[0] - measurement_time_s),
        )
        pose_skew_s = abs(sample_time_s - measurement_time_s)
        if pose_skew_s > self.target_pose_max_skew:
            self.get_logger().warn(
                "目标图像与最近飞机位姿相差 %.3fs，已忽略该观测。" % pose_skew_s,
                throttle_duration_sec=2,
            )
            return None
        return position_ned, body_to_ned, reset_state

    def _store_target_observation(
        self,
        x,
        y,
        z,
        confidence,
        measurement_time_s=None,
    ):
        """Validate one camera observation and immediately anchor it in NED."""
        if not self.is_final_aligning:
            return
        values = (x, y, z)
        if not all(math.isfinite(value) for value in values) or z <= 0.0:
            self.get_logger().warn(
                "忽略无效目标坐标: (%.3f, %.3f, %.3f)" % values,
                throttle_duration_sec=2,
            )
            return
        if confidence is not None and not math.isfinite(confidence):
            self.get_logger().warn("忽略置信度无效的目标观测。", throttle_duration_sec=2)
            return

        now = self.get_clock().now()
        now_s = now.nanoseconds / 1e9
        observation_time_s = (
            now_s if measurement_time_s is None else float(measurement_time_s)
        )
        if (
            self.final_alignment_started_at_s is not None
            and observation_time_s < self.final_alignment_started_at_s
        ):
            self.get_logger().warn(
                "忽略进入最终对准前拍摄的旧目标图像。",
                throttle_duration_sec=2,
            )
            return

        pose = self._target_pose_at(measurement_time_s)
        camera_point = np.array([x, y, z, 1.0], dtype=float)
        if pose is None:
            self.get_logger().warn(
                "没有与目标图像同步的有效飞机位姿，暂不接收该观测。",
                throttle_duration_sec=2,
            )
            return
        position_ned, body_to_ned, reset_state = pose

        target_body = self.T_body_cam @ camera_point
        target_ned = body_to_ned @ target_body[:3] + position_ned
        if not np.all(np.isfinite(target_ned)):
            self.get_logger().warn(
                "目标坐标转换到NED后无效，已忽略。",
                throttle_duration_sec=2,
            )
            return
        receive_gap_s = None
        if self.last_target_update_time is not None:
            receive_gap_s = (
                now - self.last_target_update_time
            ).nanoseconds / 1e9
            if receive_gap_s < 0.0:
                self.target_anchor_tracker.reset()
                self.last_target_measurement_time_s = None
                self.target_stream_discontinuity_pending = True
            elif receive_gap_s > self.target_timeout_duration:
                self.target_anchor_tracker.reset()
                self.target_stream_discontinuity_pending = True

        if self.last_target_measurement_time_s is not None:
            measurement_gap_s = (
                observation_time_s - self.last_target_measurement_time_s
            )
            if measurement_gap_s <= 0.0:
                self.get_logger().warn(
                    "忽略时间戳倒序或重复的目标观测。",
                    throttle_duration_sec=2,
                )
                return
            if measurement_gap_s > self.target_timeout_duration:
                self.target_anchor_tracker.reset()
                self.target_stream_discontinuity_pending = True

        if (
            self.target_anchor_reset_state is not None
            and reset_state != self.target_anchor_reset_state
        ):
            self.target_anchor_tracker.reset()
            self.target_stream_discontinuity_pending = True
        old_anchor = self.target_anchor_tracker.anchor_ned
        self.target_anchor_tracker.add_observation(
            target_ned,
            observed_at_s=observation_time_s,
            confidence=confidence,
        )
        self.target_anchor_reset_state = reset_state
        new_anchor = self.target_anchor_tracker.anchor_ned
        if old_anchor is not None and new_anchor is not None:
            anchor_jump = math.hypot(
                new_anchor[0] - old_anchor[0],
                new_anchor[1] - old_anchor[1],
            )
            if anchor_jump >= self.second_alignment_checker.threshold:
                self.target_anchor_jump_pending = True

        self.target_position = Point(x=float(x), y=float(y), z=float(z))
        self.target_confidence = None if confidence is None else float(confidence)
        self.last_target_update_time = now
        self.last_target_measurement_time_s = observation_time_s
        self.target_observation_sequence += 1
        if confidence is not None:
            self.last_confident_target_update_time = now

    def target_observation_callback(self, msg: PointCloud):
        """Receive one stamped camera point with a confidence channel."""
        if msg.header.frame_id != self.target_observation_frame_id:
            self.get_logger().warn(
                "忽略坐标系不匹配的目标观测: '%s'（期望 '%s'）。"
                % (msg.header.frame_id, self.target_observation_frame_id),
                throttle_duration_sec=2,
            )
            return
        confidence_channel = next(
            (channel for channel in msg.channels if channel.name == 'confidence'),
            None,
        )
        if (
            len(msg.points) != 1
            or confidence_channel is None
            or len(confidence_channel.values) != 1
        ):
            self.get_logger().warn(
                "/target_observation 必须包含一个点和一个confidence值。",
                throttle_duration_sec=2,
            )
            return
        point = msg.points[0]
        measurement_time_s = (
            float(msg.header.stamp.sec)
            + float(msg.header.stamp.nanosec) / 1e9
        )
        self._store_target_observation(
            point.x,
            point.y,
            point.z,
            confidence_channel.values[0],
            measurement_time_s=measurement_time_s,
        )

    def target_position_callback(self, msg: Point):
        """Fallback for detectors that only publish the legacy Point topic."""
        now = self.get_clock().now()
        if self.last_confident_target_update_time is not None:
            structured_age_s = (
                now - self.last_confident_target_update_time
            ).nanoseconds / 1e9
            # The new detector publishes the legacy Point immediately after the
            # confidence-bearing observation.  Ignore that duplicate while the
            # structured stream is alive; fall back within one missed period.
            if (
                0.0 <= structured_age_s
                <= min(0.75, self.target_timeout_duration)
            ):
                return
        self._store_target_observation(msg.x, msg.y, msg.z, confidence=None)

    def fly_to_position(self, x, y, z):
        """Fly to the specified position."""
        self.publish_position_setpoint(x, y, z)

    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        self.vehicle_local_position = vehicle_local_position
        current_reset_state = self._local_position_reset_state()
        if (
            self.is_final_aligning
            and self.target_anchor_reset_state is not None
            and current_reset_state != self.target_anchor_reset_state
        ):
            z_reference_changed = (
                current_reset_state[1] != self.target_anchor_reset_state[1]
            )
            self._invalidate_target_anchor(
                "PX4位置回调检测到本地NED参考重置：立即丢弃旧目标锚点。",
                rebase_height=z_reference_changed,
            )
        self._record_target_pose_sample()

    def _capture_widecam_pose_snapshot(self):
        """Capture a pose that is safe to associate with one camera frame."""
        position = self.vehicle_local_position
        values = (
            position.x,
            position.y,
            position.z,
            position.heading,
            self.vehicle_roll,
            self.vehicle_pitch,
        )
        if not all(math.isfinite(value) for value in values):
            return None
        if not getattr(position, 'xy_valid', True) or not getattr(position, 'z_valid', True):
            return None
        if not getattr(position, 'heading_good_for_control', True):
            return None
        if not getattr(position, 'xy_global', True) and not self.widecam_local_reference_warned:
            self.get_logger().warn(
                "PX4 xy_global is false: wide-camera targets are valid in the EKF local NED frame only."
            )
            self.widecam_local_reference_warned = True

        timestamp_us = (
            getattr(position, 'timestamp_sample', 0) or getattr(position, 'timestamp', 0)
        )
        if self.vehicle_attitude_timestamp_us is None:
            return None
        if (
            timestamp_us and
            abs(timestamp_us - self.vehicle_attitude_timestamp_us) > self.widecam_pose_attitude_skew_us
        ):
            return None

        vx = getattr(position, 'vx', float('nan'))
        vy = getattr(position, 'vy', float('nan'))
        horizontal_speed = math.hypot(vx, vy) if math.isfinite(vx) and math.isfinite(vy) else None
        return {
            'x': float(position.x),
            'y': float(position.y),
            'z': float(position.z),
            'yaw': float(position.heading),
            'roll': float(self.vehicle_roll),
            'pitch': float(self.vehicle_pitch),
            'horizontal_speed': horizontal_speed,
            'timestamp_us': timestamp_us,
            'reset_state': (
                getattr(position, 'xy_reset_counter', 0),
                getattr(position, 'z_reset_counter', 0),
                getattr(position, 'heading_reset_counter', 0),
            ),
        }

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status

    def vehicle_land_detected_callback(self, vehicle_land_detected):
        """Callback function for vehicle_land_detected topic subscriber."""
        self.vehicle_land_detected = vehicle_land_detected

    def vehicle_odometry_callback(self, msg: VehicleOdometry):
        """Callback to get the drone's full attitude (roll, pitch, yaw)."""
        # PX4 odometry msg.q is [w, x, y, z]
        # Scipy Rotation needs [x, y, z, w]
        q = np.array([msg.q[1], msg.q[2], msg.q[3], msg.q[0]], dtype=float)
        if not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-6:
            return
        q /= np.linalg.norm(q)
        
        # 从四元数转换为欧拉角 (roll, pitch, yaw)，单位是弧度
        (self.vehicle_roll, 
         self.vehicle_pitch, 
         _) = R.from_quat(q).as_euler('xyz', degrees=False)
        self.vehicle_attitude_timestamp_us = (
            getattr(msg, 'timestamp_sample', 0) or getattr(msg, 'timestamp', 0)
        )
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

    def publish_mission_state(self):
        """Publish the current high-level mission state for logging/debug tools."""
        msg = String()
        description = MISSION_STATE_DESCRIPTIONS.get(self.mission_state, "未知阶段")
        msg.data = f"{self.mission_state.name}|{description}"
        self.mission_state_publisher.publish(msg)

    def request_rtl_once(self):
        """Request RTL once, then stop owning the flight mode from this node."""
        if not self.rtl_request_sent:
            self.return_to_launch()
            self.rtl_request_sent = True
            self.get_logger().info("已发送一次 RTL 请求，停止 Offboard 模式命令与轨迹控制。")

    def handle_rtl_state(self) -> bool:
        """
        Let PX4 own RTL after mission completion or failsafe-triggered RTL.
        Returns True when the control loop should skip all Offboard outputs.
        """
        nav_state = self.vehicle_status.nav_state

        if nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_RTL and self.mission_state not in (
            MissionState.REQUEST_RTL,
            MissionState.RTL_ACTIVE,
            MissionState.DONE,
        ):
            self.mission_state = MissionState.RTL_ACTIVE
            self.rtl_request_sent = True
            self.get_logger().warn("检测到 PX4 已进入 AUTO_RTL，停止 Offboard 重试并交由 PX4 返航。")

        if self.mission_state == MissionState.REQUEST_RTL:
            self.request_rtl_once()
            if nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_RTL:
                self.mission_state = MissionState.RTL_ACTIVE
                self.rtl_active_logged = False
            else:
                self.get_logger().info(
                    f"等待 PX4 进入 AUTO_RTL，当前 nav_state={nav_state}；不再重发 Offboard。",
                    throttle_duration_sec=2,
                )
            return True

        if self.mission_state == MissionState.RTL_ACTIVE:
            if not self.rtl_active_logged:
                self.get_logger().info("PX4 RTL 已接管；等待返航/降落完成。")
                self.rtl_active_logged = True

            if (
                self.vehicle_land_detected.landed
                or self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED
            ):
                self.mission_state = MissionState.DONE
                self.get_logger().info("检测到已降落或已解锁，任务状态切换为 DONE。")
            return True

        if self.mission_state == MissionState.DONE:
            if not self.done_logged:
                self.get_logger().info("任务已完成，保持静默，不再发送飞控指令。")
                self.done_logged = True
            return True

        return False

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

    def offboard_heartbeat_timer_callback(self):
        """Keep the PX4 Offboard proof-of-life independent of slow YOLO inference."""
        if self.offboard_heartbeat_enabled:
            self.publish_offboard_control_heartbeat_signal()

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

    def _body_to_ned_rotation(self):
        """Return the current body-FRD to local-NED rotation, or ``None``."""
        values = (
            self.vehicle_roll,
            self.vehicle_pitch,
            self.vehicle_local_position.heading,
        )
        if (
            not all(math.isfinite(value) for value in values)
            or not getattr(
                self.vehicle_local_position,
                'heading_good_for_control',
                True,
            )
        ):
            return None
        return R.from_euler('xyz', values).as_matrix()

    def _local_position_reset_state(self):
        """Return PX4 counters identifying the current local NED reference."""
        position = self.vehicle_local_position
        return (
            int(getattr(position, 'xy_reset_counter', 0)),
            int(getattr(position, 'z_reset_counter', 0)),
            int(getattr(position, 'heading_reset_counter', 0)),
        )

    def _current_dropper_ned(self):
        """Return the dropper's current absolute NED position."""
        body_to_ned = self._body_to_ned_rotation()
        position = self.vehicle_local_position
        position_ned = np.array([
            position.x,
            position.y,
            position.z,
        ], dtype=float)
        position_valid = (
            getattr(position, 'xy_valid', True)
            and getattr(position, 'z_valid', True)
        )
        if (
            body_to_ned is None
            or not position_valid
            or not np.all(np.isfinite(position_ned))
        ):
            return None
        return body_to_ned @ self.p_dropper_in_body_h[:3] + position_ned

    def _reset_active_alignment_tracking(self, reason, reset_timeout=False):
        """Discard stability/PID history after target loss or a large jump."""
        if self.first_alignment_complete:
            self.second_alignment_checker.reset()
            if reset_timeout:
                self.second_align_start_timestamp = None
                self.second_align_accumulated_s = 0.0
            else:
                self._pause_alignment_timeout('second')
            self.reached_align_height = False
        else:
            self.first_alignment_checker.reset()
            if reset_timeout:
                self.first_align_start_timestamp = None
                self.first_align_accumulated_s = 0.0
            else:
                self._pause_alignment_timeout('first')
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.last_error_x = 0.0
        self.last_error_y = 0.0
        self.get_logger().warn(reason)

    def _pause_alignment_timeout(self, stage):
        """Pause a stage timeout while preserving accumulated eligible time."""
        if stage == 'first':
            timestamp_name = 'first_align_start_timestamp'
            accumulated_name = 'first_align_accumulated_s'
        elif stage == 'second':
            timestamp_name = 'second_align_start_timestamp'
            accumulated_name = 'second_align_accumulated_s'
        else:
            raise ValueError("stage must be 'first' or 'second'")

        started_at = getattr(self, timestamp_name)
        if started_at is not None:
            elapsed_s = (
                self.get_clock().now() - started_at
            ).nanoseconds / 1e9
            if math.isfinite(elapsed_s) and elapsed_s >= 0.0:
                setattr(
                    self,
                    accumulated_name,
                    getattr(self, accumulated_name) + elapsed_s,
                )
            else:
                setattr(self, accumulated_name, 0.0)
        setattr(self, timestamp_name, None)

    def _invalidate_target_anchor(self, reason, rebase_height=False):
        """Discard an anchor whose clock or local-NED reference is no longer valid."""
        position = self.vehicle_local_position
        position_ned = (float(position.x), float(position.y), float(position.z))
        position_valid = (
            getattr(position, 'xy_valid', True)
            and getattr(position, 'z_valid', True)
            and all(math.isfinite(value) for value in position_ned)
        )
        if position_valid:
            self.last_found_x_NED, self.last_found_y_NED, self.last_found_z_NED = (
                position_ned
            )
            if rebase_height and self.takeoff_target_height is not None:
                if self.first_alignment_complete:
                    self.takeoff_target_height = (
                        position_ned[2] - self.afterAlign_descentHeight
                    )
                else:
                    self.takeoff_target_height = position_ned[2]

        self.target_position = None
        self.target_confidence = None
        self.last_target_update_time = None
        self.last_confident_target_update_time = None
        self.last_target_measurement_time_s = None
        self.target_anchor_tracker.reset()
        self.target_anchor_jump_pending = False
        self.target_stream_discontinuity_pending = False
        self.target_anchor_reset_state = self._local_position_reset_state()
        self.alignment_target_was_fresh = False
        self.final_alignment_started_at_s = (
            self.get_clock().now().nanoseconds / 1e9
        )
        self.reached_align_height = False
        self._reset_active_alignment_tracking(reason, reset_timeout=True)

    def _latest_depth_target_and_dropper_world(self, require_fresh=True):
        """Return the selected fixed target anchor and current dropper in NED.

        The target was transformed once when its observation arrived.  Only the
        dropper is recomputed from the current vehicle pose, so release scoring
        cannot make a stale camera-relative point move with the aircraft.
        """
        if (
            self.target_anchor_tracker.anchor_ned is None
            or self.target_anchor_tracker.anchor_observed_at_s is None
        ):
            return None

        now_s = self.get_clock().now().nanoseconds / 1e9
        target_age_s = now_s - self.target_anchor_tracker.anchor_observed_at_s
        if not math.isfinite(target_age_s) or target_age_s < 0.0:
            return None
        stream_age_s = self.target_anchor_tracker.latest_observation_age_s(now_s)
        if require_fresh and stream_age_s > self.sim_drop_target_max_age:
            return None

        target_ned = np.asarray(self.target_anchor_tracker.anchor_ned, dtype=float)
        dropper_ned = self._current_dropper_ned()
        if dropper_ned is None:
            return None
        if not np.all(np.isfinite(target_ned)) or not np.all(np.isfinite(dropper_ned)):
            return None

        return target_ned, dropper_ned, target_age_s

    def _current_target_name(self):
        if 0 <= self.current_target_index < len(self.mission_targets_ned):
            return self.mission_targets_ned[self.current_target_index].get('name', 'UNKNOWN')
        return 'UNKNOWN'

    def _evaluate_simulated_drop(self, drop_number, reason):
        """Evaluate one virtual payload release and persist a reproducible row."""
        target_name = self._current_target_name()
        target_data = self._latest_depth_target_and_dropper_world(require_fresh=True)
        target_age_s = None
        ground_plane_source = 'unavailable'
        ground_down_m = None

        if target_data is None:
            result = self.drop_evaluator.unavailable(
                'NO_FRESH_DEPTH_TARGET',
                'No fresh depth-camera target was available at the virtual release instant.',
            )
        else:
            target_ned, dropper_ned, target_age_s = target_data
            velocity_ned = np.array([
                self.vehicle_local_position.vx,
                self.vehicle_local_position.vy,
                self.vehicle_local_position.vz,
            ], dtype=float)
            if not np.all(np.isfinite(velocity_ned)):
                result = self.drop_evaluator.unavailable(
                    'NO_VALID_VELOCITY',
                    'PX4 local NED velocity was invalid at the virtual release instant.',
                )
            else:
                if self.sim_drop_ground_z is None:
                    ground_down_m = float(target_ned[2])
                    ground_plane_source = 'depth_target'
                else:
                    ground_down_m = self.sim_drop_ground_z
                    ground_plane_source = 'configured'
                result = self.drop_evaluator.evaluate(
                    dropper_ned,
                    velocity_ned,
                    target_ned,
                    ground_down_m,
                )

        row = {
            'timestamp_sec': f'{time.time():.6f}',
            'drop_number': drop_number,
            'reason': reason,
            'target_name': target_name,
            'target_age_s': '' if target_age_s is None else f'{target_age_s:.6f}',
            'alignment_complete': self.second_alignment_complete,
            'ground_plane_source': ground_plane_source,
            'ground_down_m': '' if ground_down_m is None else f'{ground_down_m:.6f}',
            **result.as_dict(),
        }
        self.drop_eval_writer.writerow(row)
        self.drop_eval_file.flush()
        self.drop_evaluations[drop_number] = row
        self.latest_drop_evaluation = row

        radial_error_text = (
            'nan'
            if result.radial_error_m is None
            else f'{result.radial_error_m:.3f}'
        )
        result_message = String()
        result_message.data = (
            f'drop={drop_number}|target={target_name}|reason={reason}|status={result.status}|'
            f'hit={result.hit}|radial_error_m={radial_error_text}'
        )
        self.simulated_drop_result_publisher.publish(result_message)
        if result.status == 'HIT':
            self.get_logger().info(
                f"虚拟投放 #{drop_number} 命中 '{target_name}'："
                f"落点误差 {result.radial_error_m:.3f}m <= {result.hit_radius_m:.3f}m。"
            )
        elif result.status == 'MISS':
            self.get_logger().warn(
                f"虚拟投放 #{drop_number} 未命中 '{target_name}'："
                f"落点误差 {result.radial_error_m:.3f}m > {result.hit_radius_m:.3f}m。"
            )
        else:
            self.get_logger().warn(
                f"虚拟投放 #{drop_number} 无法判定精确落点：{result.status}。"
            )

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
        if hasattr(self, 'drop_eval_file') and not self.drop_eval_file.closed:
            self.drop_eval_file.close()
            self.get_logger().info("虚拟投放评估日志已关闭。")
        # # 清理摄像头
        # if self.cap and self.cap.isOpened():
        #     self.cap.release()
        # 关闭所有OpenCV窗口
        if self.show_video:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
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
 
    
    def drop_payload(self, drop_number: int, reason: str = 'unspecified'):
        """
        启动指定编号的多步骤投水序列。
        这个函数只负责启动，不负责管理过程。
        """
        if self.current_dropping_state[drop_number] == DroppingState.IDLE:
            self.get_logger().info(f"启动第 {drop_number} 次投水序列...")
            if self.drop_eval_enabled:
                self._evaluate_simulated_drop(drop_number, reason)
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
            self._pause_alignment_timeout('first')
            self.second_align_start_timestamp = None
            self.second_align_accumulated_s = 0.0
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
            self._pause_alignment_timeout('second')
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

    def coordinate_current_FRD2NED(self, x, y, origin_x=None, origin_y=None, yaw=None):
        """将当前机体 FRD 平面坐标转换为世界 NED 坐标。

        视觉测量值相对于拍摄该帧时的机体；因此平移和旋转都必须使用
        当前的局部位置与航向，而不是固定使用起飞点或投放区的位置。
        """
        if origin_x is None:
            origin_x = self.vehicle_local_position.x
        if origin_y is None:
            origin_y = self.vehicle_local_position.y
        if yaw is None:
            yaw = self.vehicle_local_position.heading

        x_target = x * math.cos(yaw) - y * math.sin(yaw) + origin_x
        y_target = x * math.sin(yaw) + y * math.cos(yaw) + origin_y
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
        self.first_align_accumulated_s = 0.0
        self.second_align_accumulated_s = 0.0
        self.target_position = None
        self.target_confidence = None
        self.last_target_update_time = None
        self.last_confident_target_update_time = None
        self.last_target_measurement_time_s = None
        self.final_alignment_started_at_s = None
        self.target_anchor_tracker.reset()
        self.target_anchor_jump_pending = False
        self.target_stream_discontinuity_pending = False
        self.target_anchor_reset_state = None
        self.alignment_target_was_fresh = False
        self.last_logged_target_observation_sequence = -1
        self.last_found_x_NED = None
        self.last_found_y_NED = None
        self.last_found_z_NED = None
        self.reached_align_height = False
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.last_error_x = 0.0
        self.last_error_y = 0.0
        
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

    def _begin_final_alignment(self, hold_position_ned):
        """Start a clean alignment session and hold the planned approach point."""
        hold_position_ned = tuple(float(value) for value in hold_position_ned)
        if (
            len(hold_position_ned) != 3
            or not all(math.isfinite(value) for value in hold_position_ned)
        ):
            raise ValueError("hold_position_ned must contain three finite values")

        self.first_alignment_complete = False
        self.second_alignment_complete = False
        self.first_alignment_checker.reset()
        self.second_alignment_checker.reset()
        self.first_align_start_timestamp = None
        self.second_align_start_timestamp = None
        self.first_align_accumulated_s = 0.0
        self.second_align_accumulated_s = 0.0
        self.reached_align_height = False
        self.target_position = None
        self.target_confidence = None
        self.last_target_update_time = None
        self.last_confident_target_update_time = None
        self.last_target_measurement_time_s = None
        self.target_anchor_tracker.reset()
        self.target_anchor_jump_pending = False
        self.target_stream_discontinuity_pending = False
        self.target_anchor_reset_state = self._local_position_reset_state()
        self.alignment_target_was_fresh = False
        self.last_logged_target_observation_sequence = (
            self.target_observation_sequence
        )
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.last_error_x = 0.0
        self.last_error_y = 0.0
        self.last_found_x_NED, self.last_found_y_NED, self.last_found_z_NED = (
            hold_position_ned
        )
        self.final_alignment_started_at_s = (
            self.get_clock().now().nanoseconds / 1e9
        )
        self.is_navigating_to_target = False
        self.is_final_aligning = True

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
        now = self.get_clock().now()
        now_s = now.nanoseconds / 1e9

        current_reset_state = self._local_position_reset_state()
        if (
            self.target_anchor_reset_state is not None
            and current_reset_state != self.target_anchor_reset_state
        ):
            z_reference_changed = (
                current_reset_state[1] != self.target_anchor_reset_state[1]
            )
            self._invalidate_target_anchor(
                "PX4本地位置或航向参考已重置：丢弃旧NED目标并等待新图像。",
                rebase_height=z_reference_changed,
            )

        is_target_valid = False
        receive_age_s = math.inf
        measurement_age_s = self.target_anchor_tracker.latest_observation_age_s(
            now_s
        )
        if self.target_position and self.last_target_update_time:
            receive_age_s = (
                now - self.last_target_update_time
            ).nanoseconds / 1e9
            if receive_age_s < 0.0:
                self._invalidate_target_anchor(
                    "ROS时钟回拨：丢弃旧目标并重新开始对准。"
                )
                measurement_age_s = math.inf
            elif (
                receive_age_s < self.target_timeout_duration
                and measurement_age_s < self.target_timeout_duration
            ):
                is_target_valid = True
            elif self.log_counter % 25 == 0:
                self.get_logger().warn(
                    "目标信息已超时（接收年龄=%.2fs，图像年龄=%.2fs，阈值=%.2fs）。"
                    % (
                        receive_age_s,
                        measurement_age_s,
                        self.target_timeout_duration,
                    )
                )

        old_anchor = self.target_anchor_tracker.anchor_ned
        self.target_anchor_tracker.refresh(now_s)
        new_anchor = self.target_anchor_tracker.anchor_ned
        if old_anchor is not None and new_anchor is not None and old_anchor != new_anchor:
            anchor_jump = math.hypot(
                new_anchor[0] - old_anchor[0],
                new_anchor[1] - old_anchor[1],
            )
            if anchor_jump >= self.second_alignment_checker.threshold:
                self.target_anchor_jump_pending = True

        p_dropper_in_world = self._current_dropper_ned()
        has_active_anchor = (
            self.target_anchor_tracker.is_active(now_s)
            and p_dropper_in_world is not None
        )
        alignment_data_valid = is_target_valid and has_active_anchor

        if self.target_stream_discontinuity_pending:
            self._reset_active_alignment_tracking(
                "目标观测时间不连续：重新开始连续对准计时。",
                reset_timeout=True,
            )
            self.target_stream_discontinuity_pending = False
            self.alignment_target_was_fresh = False
            self.target_anchor_jump_pending = False
        elif self.alignment_target_was_fresh and not alignment_data_valid:
            self._reset_active_alignment_tracking(
                "目标观测中断：重置连续对准计时，但短时继续追踪固定NED锚点。"
            )
            self.target_anchor_jump_pending = False
        elif alignment_data_valid and self.target_anchor_jump_pending:
            self._reset_active_alignment_tracking(
                "最高置信度目标锚点发生明显变化：重新开始连续对准计时。",
                reset_timeout=True,
            )
            self.target_anchor_jump_pending = False
        self.alignment_target_was_fresh = alignment_data_valid
                    
        is_in_second_alignment = self.first_alignment_complete and not self.second_alignment_complete
        is_in_first_alignment = not self.first_alignment_complete

        first_target_z = self.takeoff_target_height
        second_target_z = (
            None
            if self.takeoff_target_height is None
            else self.takeoff_target_height + self.afterAlign_descentHeight
        )
        first_altitude_ok = altitude_within_threshold(
            self.vehicle_local_position.z,
            first_target_z,
            self.alignment_altitude_threshold,
        )
        second_altitude_ok = altitude_within_threshold(
            self.vehicle_local_position.z,
            second_target_z,
            self.alignment_altitude_threshold,
        )
        first_stage_eligible = alignment_data_valid and first_altitude_ok
        second_stage_eligible = alignment_data_valid and second_altitude_ok

        if is_in_first_alignment and not first_stage_eligible:
            if self.first_align_start_timestamp is not None:
                self.first_alignment_checker.reset()
            self._pause_alignment_timeout('first')
        if is_in_second_alignment and not second_stage_eligible:
            self._pause_alignment_timeout('second')

        if is_in_first_alignment and first_stage_eligible:
            # 启动计时器 (如果尚未启动)
            if self.first_align_start_timestamp is None:
                self.first_align_start_timestamp = self.get_clock().now()
                self.get_logger().info(
                    "第一次对准有效计时继续：已累计 %.2fs / %.2fs。"
                    % (self.first_align_accumulated_s, self.first_align_maxtime)
                )

            # 计算已过时间
            elapsed_first_align_time = self.first_align_accumulated_s + (
                self.get_clock().now() - self.first_align_start_timestamp
            ).nanoseconds / 1e9

            # 检查是否超时
            if elapsed_first_align_time > self.first_align_maxtime:
                if self.current_dropping_state[1] == DroppingState.IDLE and not self.Is_Finish_1st_Drop:
                    self.drop_payload(1, reason='first_alignment_timeout') # 启动第一次投水
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
                    self.drop_payload(2, reason='first_alignment_timeout') # 启动第二次投水
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
        
        # 第二阶段只有在新鲜目标和有效飞机位姿同时存在时才累计超时。
        if is_in_second_alignment and second_stage_eligible:
            # 启动计时器 (如果尚未启动)
            if self.second_align_start_timestamp is None:
                self.second_align_start_timestamp = self.get_clock().now()
                self.get_logger().info(
                    "第二次对准有效计时继续：已累计 %.2fs / %.2fs。"
                    % (self.second_align_accumulated_s, self.second_align_maxtime)
                )

            # 计算已过时间
            elapsed_drop_time = self.second_align_accumulated_s + (
                self.get_clock().now() - self.second_align_start_timestamp
            ).nanoseconds / 1e9
            
            # 检查是否超时
            if elapsed_drop_time > self.second_align_maxtime:
                if self.current_dropping_state[1] == DroppingState.IDLE and not self.Is_Finish_1st_Drop:
                    self.drop_payload(1, reason='second_alignment_timeout') # 启动第一次投水
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
                    self.drop_payload(2, reason='second_alignment_timeout') # 启动第二次投水
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
            
      
        if has_active_anchor:
            # 每条新观测只记录一次；控制循环使用固定的世界坐标锚点。
            if (
                alignment_data_valid
                and self.target_observation_sequence
                != self.last_logged_target_observation_sequence
            ):
                self.pixel_log_writer.writerow([
                    time.time(),
                    self.target_position.x,
                    self.target_position.y,
                    self.target_confidence,
                    "second" if self.first_alignment_complete else "first",
                ])
                self.last_logged_target_observation_sequence = (
                    self.target_observation_sequence
                )

            target_anchor_ned = np.asarray(
                self.target_anchor_tracker.anchor_ned,
                dtype=float,
            )
            error_ned = target_anchor_ned - p_dropper_in_world

            # 将NED世界误差向量转换为当前机体FRD误差，供PID使用。
            error_frd_x, error_frd_y = self.coordinate_NED2FRD_vector(
                error_ned[0],
                error_ned[1],
            )

            if self.log_counter % 25 == 0:
                confidence_text = (
                    "legacy"
                    if self.target_anchor_tracker.anchor_confidence is None
                    else f"{self.target_anchor_tracker.anchor_confidence:.3f}"
                )
                self.get_logger().info(
                    "追踪固定NED目标: (%.3f, %.3f), confidence=%s, fresh=%s"
                    % (
                        target_anchor_ned[0],
                        target_anchor_ned[1],
                        confidence_text,
                        alignment_data_valid,
                    )
                )

            # === 2. 将精确误差 "喂" 给你的PID控制器 ===
            distance = math.hypot(error_frd_x, error_frd_y)
            
            if distance < self.epsilon and alignment_data_valid:
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
            elif distance < self.epsilon:
                # 短时丢帧只使用固定锚点的几何误差，避免旧观测期间积分累积。
                control_x = error_frd_x
                control_y = error_frd_y
                self.integral_x = 0.0
                self.integral_y = 0.0
                self.last_error_x = 0.0
                self.last_error_y = 0.0
            else:
                # ——————— 大误差阶段：饱和P控制 ———————
                if self.log_counter % 25 == 0:
                    self.get_logger().info(f"饱和P控制阶段 - 误差:{distance:.3f}m >= 阈值:{self.epsilon:.3f}m")
                
                # 📌 饱和比例控制
                scale = self.align_maxstep / distance
                control_x = error_frd_x * scale
                control_y = error_frd_y * scale
                self.integral_x, self.integral_y, self.last_error_x, self.last_error_y = 0.0, 0.0, 0.0, 0.0

            # PID输出属于当前机体FRD，必须按当前航向转回NED。
            target_x_NED, target_y_NED = self.coordinate_current_FRD2NED(
                control_x,
                control_y,
            )
            
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
                if first_stage_eligible:
                    self.first_alignment_check(
                        precise_target_x_NED,
                        precise_target_y_NED,
                    )
                self.last_found_x_NED = target_x_NED
                self.last_found_y_NED = target_y_NED
                self.last_found_z_NED = self.takeoff_target_height

            elif self.first_alignment_complete and not self.second_alignment_complete:
                if self.log_counter % 25 == 0:
                    self.get_logger().info("执行第二次精确对准")
                self.fly_to_position(target_x_NED, target_y_NED, second_target_z)

                # <<< 新增：高度门控 >>>
                target_z = second_target_z
                if not second_altitude_ok:
                    if self.reached_align_height:
                        self.second_alignment_checker.reset()
                        self.get_logger().warn(
                            "第二次对准高度超差：重置连续水平对准计时。"
                        )
                    self.reached_align_height = False
                elif not self.reached_align_height:
                    self.reached_align_height = True
                    self.second_alignment_checker.reset()
                    self.get_logger().warn(
                        "高度达到，开始检查第二次水平对准精度。"
                    )
                elif second_stage_eligible:
                    self.second_alignment_check(
                        precise_target_x_NED,
                        precise_target_y_NED,
                    )

                self.last_found_x_NED = target_x_NED
                self.last_found_y_NED = target_y_NED
                self.last_found_z_NED = self.takeoff_target_height + self.afterAlign_descentHeight
            
            # ============== 投水逻辑 ==============
            if self.first_alignment_complete and self.second_alignment_complete and not self.is_drop_initiated_for_current_target:
    # 只负责启动，不设置完成标志
                if self.current_dropping_state[1] == DroppingState.IDLE and not self.Is_Finish_1st_Drop:
                    self.drop_payload(1, reason='precise_alignment') # 启动第一次投水
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
                    self.drop_payload(2, reason='precise_alignment') # 启动第二次投水
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
            # 锚点过期后冻结最后一个绝对NED设定点，避免“原地等待”随风漂移。
            if all(value is not None for value in (
                self.last_found_x_NED,
                self.last_found_y_NED,
                self.last_found_z_NED,
            )):
                if self.log_counter % 25 == 0:
                    self.get_logger().info("目标锚点已过期，保持最后绝对NED设定点。")
                self.fly_to_position(self.last_found_x_NED, self.last_found_y_NED, self.last_found_z_NED)
            else:
                if self.log_counter % 25 == 0:
                    self.get_logger().info("无目标记录，原地等待")
                hold_z = (
                    self.takeoff_target_height + self.afterAlign_descentHeight
                    if self.first_alignment_complete
                    else self.takeoff_target_height
                )
                self.fly_to_position(
                    self.vehicle_local_position.x,
                    self.vehicle_local_position.y,
                    hold_z,
                )
    
    
    def _collect_global_search_map_sample(self, vision_info, sample_pose):
        """Convert one frame's targets to NED using the pose captured with it."""
        if not vision_info or sample_pose is None:
            return

        reset_state = sample_pose['reset_state']
        if self.widecam_map_reset_state is None:
            self.widecam_map_reset_state = reset_state
        elif reset_state != self.widecam_map_reset_state:
            self.get_logger().warn(
                "PX4 local-position or heading reset during wide-camera mapping; "
                "discarding earlier map samples."
            )
            self.map_data_collection.clear()
            self.world_target_coordinates_ned.clear()
            self.widecam_map_reset_state = reset_state

        horizontal_speed = sample_pose.get('horizontal_speed')
        if horizontal_speed is not None and horizontal_speed > self.widecam_map_max_speed:
            if self.log_counter % 25 == 0:
                self.get_logger().info(
                    "Skip wide-camera map sample while moving: %.2fm/s > %.2fm/s" % (
                        horizontal_speed, self.widecam_map_max_speed
                    )
                )
            return

        frame_map = {}
        for target in vision_info:
            name = target.get('name')
            if name not in ("Left", "Middle", "Right") or 'coords_frd' not in target:
                continue
            x_frd, y_frd = target['coords_frd']
            if not math.isfinite(x_frd) or not math.isfinite(y_frd):
                continue
            frame_map[name] = self.coordinate_current_FRD2NED(
                x_frd,
                y_frd,
                origin_x=sample_pose['x'],
                origin_y=sample_pose['y'],
                yaw=sample_pose['yaw'],
            )

        if frame_map:
            self.map_data_collection.append(frame_map)

    def _calculate_and_store_average_map(self):
        """
        计算收集到的多帧地图数据的平均值，并将其存储到最终的NED坐标地图中。
        """
        self.mission_targets_ned.clear()
        self.world_target_coordinates_ned.clear()

        if not self.map_data_collection:
            self.get_logger().error("无法计算平均地图，因为没有收集到数据。")
            return

        samples_by_name = {"Left": [], "Middle": [], "Right": []}
        for frame_map in self.map_data_collection:
            for name, coords in frame_map.items():
                if name in samples_by_name and all(math.isfinite(value) for value in coords):
                    samples_by_name[name].append(coords)

        inlier_counts = {"Left": 0, "Middle": 0, "Right": 0}
        for name, samples in samples_by_name.items():
            if samples:
                points = np.asarray(samples, dtype=float)
                median = np.median(points, axis=0)
                residuals = np.linalg.norm(points - median, axis=1)
                residual_median = float(np.median(residuals))
                residual_mad = float(np.median(np.abs(residuals - residual_median)))
                inlier_limit = max(
                    self.widecam_map_outlier_floor_m,
                    residual_median + 3.0 * max(residual_mad, 0.02),
                )
                inliers = points[residuals <= inlier_limit]
                if len(inliers) == 0:
                    continue

                ned_x, ned_y = np.mean(inliers, axis=0)
                inlier_counts[name] = len(inliers)
                self.world_target_coordinates_ned[name] = (ned_x, ned_y)
                self.get_logger().info(
                    f"  -> 坐标 '{name}' 使用 {len(inliers)}/{len(points)} 帧 (NED): "
                    f"({ned_x:.2f}, {ned_y:.2f}), outlier_limit={inlier_limit:.2f}m"
                )

        self.get_logger().info(
            "全局搜索目标累计样本: "
            f"Left={inlier_counts['Left']}/{len(samples_by_name['Left'])} 帧, "
            f"Middle={inlier_counts['Middle']}/{len(samples_by_name['Middle'])} 帧, "
            f"Right={inlier_counts['Right']}/{len(samples_by_name['Right'])} 帧 "
            f"(有效视觉帧总数={len(self.map_data_collection)}；同一帧可同时计入多个目标)"
        )
        self.get_logger().info(f"将按照用户指定的顺序进行打击: {self.target_priority}")

        for target_name in self.target_priority:
            if target_name in self.world_target_coordinates_ned:
                self.mission_targets_ned.append({
                    'name': target_name,
                    'coords_ned': self.world_target_coordinates_ned[target_name],
                })
                ned_x, ned_y = self.world_target_coordinates_ned[target_name]
                self.get_logger().info(
                    f"  -> 已规划平均目标 '{target_name}' @ NED({ned_x:.2f}, {ned_y:.2f})"
                )
            else:
                self.get_logger().warn(
                    f"  -> 用户指定的目标 '{target_name}' 没有足够的平均坐标，将跳过。"
                )

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
                
                ned_x, ned_y = self.coordinate_current_FRD2NED(x_frd, y_frd)
                
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
            x_frd, y_frd = target_data['coords_frd']
            ned_x, ned_y = self.coordinate_current_FRD2NED(x_frd, y_frd)
            
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
        if self.handle_rtl_state():
            self.offboard_heartbeat_enabled = False
            self.offboard_setpoint_counter += 1
            return

        if not self.is_vision_ready:
            self.get_logger().info(
                "正在等待视觉模型初始化完成，暂不发送 Offboard 心跳或模式切换。",
                throttle_duration_sec=2,
            )
            return
        
        # 更新日志计数器
        self.log_counter += 1
        
        # --- 视觉处理部分 ---
        # ret, frame = self.cap.read()
        # if not ret:
        #     self.get_logger().warn("无法捕获图像")
        #     return

        

        # +++ (以下是新的替换代码) +++
        with self._frame_lock:
            has_frame = self.latest_frame is not None
            frame_received_time = self.frame_received_time

        if not has_frame:
            self.get_logger().warn("尚未接收到任何图像帧...", throttle_duration_sec=2)
            return
        # (可选但推荐) 检查图像是否过时
        time_since_last_frame = (self.get_clock().now() - frame_received_time).nanoseconds / 1e9
        if time_since_last_frame > self.image_timeout_sec:
            self.get_logger().error(
                f"图像话题已超时（{time_since_last_frame:.2f}s > "
                f"{self.image_timeout_sec:.2f}s）！检查桥接或仿真是否正常。",
                throttle_duration_sec=2,
            )
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                self.get_logger().warn("Offboard 期间视觉流中断，主动请求 RTL。")
                self.mission_state = MissionState.REQUEST_RTL
                self.offboard_heartbeat_enabled = False
                self.request_rtl_once()
            return

        # 只有模型已就绪且相机流新鲜时，才允许 PX4 进入 Offboard。该独立
        # 心跳定时器不会被 YOLO 推理阻塞，避免触发 COM_OF_LOSS_T failsafe。
        self.offboard_heartbeat_enabled = True

        #进入offboard前发布位置控制点
             
        if self.offboard_setpoint_counter < 10:
            if self.vehicle_status.nav_state == 0 and self.offboard_setpoint_counter == 0:
                self.get_logger().warn(
                    "飞控状态尚未收到(nav_state=0)，"
                    "请确认仿真 PX4、MicroXRCEAgent 和 DDS 桥接正常。"
                )
            self.publish_position_setpoint(self.vehicle_local_position.x, self.vehicle_local_position.y, self.vehicle_local_position.z)
            self.engage_offboard_mode()  
            # 仅在日志计数满足条件时打印
            if self.log_counter % 10 == 0:
                self.get_logger().info(
                    f"尝试切入offboard(第{self.offboard_setpoint_counter + 1}次), "
                    f"==============向前飞行距离{self.forward_x}m==================="
                )

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
                    with self._map_data_lock:
                        self.map_data_collection.clear()
                        self.world_target_coordinates_ned.clear()
                        self.mission_targets_ned.clear()
                        self.widecam_map_reset_state = None
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
                    with self._vision_result_lock:
                        self.current_vision_info = list(self.latest_vision_info)
                   
                    
                    #启动全局搜索计时器
                    if self.search_start_time is None:
                        self.get_logger().info(f"进入全局搜索，将持续 {self.search_timeout}s 建立稳定跟踪...")
                        self.search_start_time = self.get_clock().now()
                    
                    self.global_search_target_z = float(self.initial_z + self.global_search_height)
                    self.publish_position_setpoint(self.DropArea_x, self.DropArea_y, self.global_search_target_z)

                    elapsed_search_time = (self.get_clock().now() - self.search_start_time).nanoseconds / 1e9
                    
                    ## 全局搜索到达时间后
                    if elapsed_search_time > self.search_timeout:
                        self.get_logger().info("搜索时间到，开始根据多帧平均结果和用户优先级构建最终任务地图。")
                        with self._map_data_lock:
                            self._calculate_and_store_average_map()
                        
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
                                self.is_smoothing_descent = False # 关闭平滑模式
                                self._begin_final_alignment((end_x, end_y, end_z))
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
                        self.drop_payload(1, reason='mission_timeout')
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
                                self.drop_payload(2, reason='mission_timeout')
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
                    with self._vision_result_lock:
                        self.current_vision_info = list(self.latest_vision_info)
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
                        self.mission_state = MissionState.REQUEST_RTL
                        self.get_logger().info("所有任务阶段均已完成，进入 REQUEST_RTL。")
                        return
        
        else:
            # 只有在过了初始的切换阶段后才打印日志，避免启动时的干扰
            if self.offboard_setpoint_counter >= 10:
                if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_RTL:
                    self.mission_state = MissionState.RTL_ACTIVE
                    self.rtl_request_sent = True
                    self.get_logger().warn("PX4 已进入 AUTO_RTL，停止 Offboard 重试。")
                    self.offboard_setpoint_counter += 1
                    return
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
        
        elasped_timer_time = (self.get_clock().now() - timer_start).nanoseconds / 1e9
        if self.offboard_setpoint_counter % 150 == 0:
            self.get_logger().info(f"控制循环花费时间：{elasped_timer_time:.5f}s")

    def display_latest_annotated_frame(self):
        """Pump OpenCV GUI events from the process main thread only."""
        if not self.show_video:
            return
        with self._vision_result_lock:
            annotated_frame = self.latest_annotated_frame
        if annotated_frame is not None:
            try:
                cv2.imshow("Drone View", annotated_frame)
                cv2.waitKey(1)
            except cv2.error as exc:
                self.show_video = False
                self.get_logger().warn(
                    f"OpenCV窗口不可用，后续自动无头运行: {exc}"
                )

    def vision_timer_callback(self):
        """
        这个回调以较低频率运行，专门处理耗时的视觉任务。
        """
        timer_start = self.get_clock().now()
        if not self.is_vision_ready:
            if self.vision_controller.load_model():
                self.is_vision_ready = True
                self.get_logger().info("视觉系统准备就绪，开始执行任务逻辑。")
            else:
                self.get_logger().error(
                    "视觉系统初始化失败，节点将不执行任务。",
                    throttle_duration_sec=2,
                )
            return

        # 只对新帧推理。重复推理同一帧会占满执行器并导致 Offboard 心跳断流。
        with self._frame_lock:
            frame_sequence = self.latest_frame_sequence
            if self.latest_frame is None or frame_sequence == self.last_processed_frame_sequence:
                return
            frame_to_process = self.latest_frame.copy()
            frame_pose = (
                dict(self.latest_frame_pose_snapshot)
                if self.latest_frame_pose_snapshot is not None else None
            )
            self.last_processed_frame_sequence = frame_sequence

        if frame_pose is None:
            return
        
        # <<< 核心决策逻辑 >>>
        num_targets_for_vision = 0 # 默认不处理
        
        # 状态1：投水前的全局搜索，需要找 3 个目标
        if self.mission_state == MissionState.GLOBAL_SEARCH:
            num_targets_for_vision = 3
        
        # 状态2：侦察阶段的搜索，需要找 5 个目标
        elif self.mission_state == MissionState.RECON_SEARCH:
            num_targets_for_vision = 5

        # 核心视觉处理调用
        initial_z = self.initial_z if self.initial_z is not None else 0.0
        current_altitude = frame_pose['z'] - initial_z
        vision_info, annotated_frame = self.vision_controller.process_frame(
            frame_to_process, 
            current_altitude,
            max_targets_to_confirm=num_targets_for_vision, # <<< 将决策结果传入
            roll=frame_pose['roll'],
            pitch=frame_pose['pitch'],
        )

        # 只有在进行有效处理时才更新视觉信息
        if num_targets_for_vision > 0:
            with self._vision_result_lock:
                self.latest_vision_info = vision_info
            if self.mission_state == MissionState.GLOBAL_SEARCH:
                with self._map_data_lock:
                    self._collect_global_search_map_sample(vision_info, frame_pose)
        
        # 更新用于显示的 annotated_frame (无论是否处理都更新，以便显示状态)
        cv2.putText(annotated_frame, f"State: {self.mission_state.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Vision Targets: {num_targets_for_vision}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2) # 添加一个状态显示
        if self.latest_drop_evaluation is not None:
            drop_status = self.latest_drop_evaluation['status']
            radial_error = self.latest_drop_evaluation['radial_error_m']
            error_text = 'n/a' if radial_error is None else f'{float(radial_error):.3f}'
            drop_color = (0, 255, 0) if drop_status == 'HIT' else (0, 0, 255)
            cv2.putText(
                annotated_frame,
                f"Last virtual drop: {drop_status}, error={error_text}m",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                drop_color,
                2,
            )
        with self._vision_result_lock:
            self.latest_annotated_frame = annotated_frame

        elasped_timer_time = (self.get_clock().now() - timer_start).nanoseconds / 1e9
        if self.offboard_setpoint_counter % 150 == 0:
            self.get_logger().info(f"视觉循环花费时间：{elasped_timer_time:.5f}")


def spin_control_node_safely(node: Node) -> None:
    """
    Keep the control node alive when a timer/subscription callback raises.
    Normal shutdown paths still leave the loop so resources can be cleaned up.
    """
    # 四个回调组分别负责任务状态机、视觉推理、图像接收和心跳。
    # ROS executor 必须持续运行在自己的调度线程中；OpenCV GUI 留在主线程。
    # 这样即使窗口事件循环短暂卡顿，也不会停止图像、控制或心跳回调的派发。
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor_stop = threading.Event()

    def spin_executor():
        while rclpy.ok() and not executor_stop.is_set():
            try:
                executor.spin_once(timeout_sec=0.1)
            except (ExternalShutdownException, ShutdownException):
                break
            except Exception as e:
                node.get_logger().error(
                    "控制回调发生异常，已拦截并继续运行，避免控制节点被关闭："
                    f"{e}\n{traceback.format_exc()}",
                    throttle_duration_sec=2.0,
                )

    executor_thread = threading.Thread(
        target=spin_executor,
        name='offboard-ros-executor',
        daemon=True,
    )
    executor_thread.start()

    try:
        while rclpy.ok() and executor_thread.is_alive():
            if node.show_video:
                node.display_latest_annotated_frame()
            time.sleep(0.01)
    finally:
        executor_stop.set()
        executor.wake()
        executor_thread.join(timeout=2.0)
        executor.shutdown()
        executor_thread.join(timeout=2.0)
        executor.remove_node(node)


def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)

    # 2. 设置我们自己的命令行参数解析器
    parser = argparse.ArgumentParser(description="Offboard control script for PX4 drone mission.")

    pkg_name = 'control'  # 替换的包名
        
    try:
        pkg_share_dir = get_package_share_directory(pkg_name)
        
        # 假设你的pt文件在包的根目录
        default_weights_path = os.path.join(pkg_share_dir, 'models', '26n_0807_bright_needle.pt')
        
        # 检查文件是否存在，不存在则使用备用路径
        if not os.path.exists(default_weights_path):
            print(f"警告：默认模型文件不存在: {default_weights_path}")
            # 可以设置为空字符串或其他默认值
            default_weights_path = ""
            
    except PackageNotFoundError:
        print(f"错误：找不到 ROS2 包 {pkg_name}")
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
    
    parser.add_argument('--forward-x', type=float, default=3,
                        help='Forward distance to fly to the drop area in meters.')
    parser.add_argument('--search-height', type=float, default=-4.5,
                        help='Global search height in meters (negative value for altitude).')
   
    parser.add_argument('--align-maxstep', type=float, default=0.2,
                        help='Maximum step size for each alignment adjustment.')
    parser.add_argument('--target-timeout-duration', type=float, default=1.2,
                        help='最新目标观测保持 fresh 的时长（秒）。')
    parser.add_argument('--target-confidence-window', type=float, default=4.0,
                        help='选择最高置信度目标观测的滚动时间窗（秒）。')
    parser.add_argument('--target-anchor-hold-duration', type=float, default=2.5,
                        help='丢失新观测后仍朝固定世界目标移动的最长时间（秒）。')
    parser.add_argument('--target-pose-max-skew', type=float, default=0.20,
                        help='目标图像时间与用于NED转换的飞机位姿最大时间差（秒）。')
    parser.add_argument('--target-pose-attitude-max-skew', type=float, default=0.10,
                        help='组成目标位姿的PX4位置与姿态样本最大时间差（秒）。')
    parser.add_argument('--target-observation-frame-id', type=str,
                        default='target_camera_optical_frame',
                        help='目标PointCloud必须使用的相机光学坐标系名称。')

    
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
    parser.add_argument('--second-align-maxtime', type=float, default=8.0,
                        help='Maximum time in seconds for each search attempt.')
    parser.add_argument('--first-align-maxtime', type=float, default=12.0,
                        help='Maximum time in seconds for the first alignment phase before forcing a drop.')
    
    
    parser.add_argument('--depthcam_xoffset', type=float, default=-0.065,
                        help='深度相机的x方向误差.')
    parser.add_argument('--depthcam_yoffset', type=float, default=0.033,
                        help='深度相机的y方向误差.')
    parser.add_argument('--depthcam-zoffset', type=float, default=0.15,
                        help='深度相机相对机体原点的下向偏移（FRD，米）。')

    parser.add_argument('--dropper-xoffset', type=float, default=0.0,
                        help='虚拟投放器相对机体原点的前向偏移（FRD，米）。')
    parser.add_argument('--dropper-yoffset', type=float, default=0.0,
                        help='虚拟投放器相对机体原点的右向偏移（FRD，米）。')
    parser.add_argument('--dropper-zoffset', type=float, default=0.15,
                        help='虚拟投放器相对机体原点的下向偏移（FRD，米）。')

    parser.add_argument('--widecam-xoffset', type=float, default=0.0,
                        help='广角相机光心相对机体原点的前向偏移（FRD，米）。')
    parser.add_argument('--widecam-yoffset', type=float, default=0.0,
                        help='广角相机光心相对机体原点的右向偏移（FRD，米）。')
    parser.add_argument('--widecam-zoffset', type=float, default=0.0,
                        help='广角相机光心相对机体原点的下向偏移（FRD，米）。')
    parser.add_argument('--widecam-roll-deg', type=float, default=0.0,
                        help='广角相机相对默认向下安装姿态的机体FRD横滚补偿角（度）。')
    parser.add_argument('--widecam-pitch-deg', type=float, default=0.0,
                        help='广角相机相对默认向下安装姿态的机体FRD俯仰补偿角（度）。')
    parser.add_argument('--widecam-yaw-deg', type=float, default=0.0,
                        help='广角相机相对默认向下安装姿态的机体FRD偏航补偿角（度）。')
    parser.add_argument('--widecam-min-ground-ray-down', type=float, default=0.15,
                        help='Minimum positive down component of a wide-camera ray used for ground intersection.')
    parser.add_argument('--widecam-max-ground-range', type=float, default=30.0,
                        help='Maximum accepted horizontal ground-intersection range in meters.')
    parser.add_argument('--widecam-map-max-speed', type=float, default=0.40,
                        help='Only collect wide-camera map samples at or below this horizontal speed (m/s).')
    parser.add_argument('--widecam-map-outlier-floor', type=float, default=0.30,
                        help='Minimum NED residual threshold for robust wide-camera map averaging (m).')
    parser.add_argument('--widecam-max-pose-attitude-skew', type=float, default=0.10,
                        help='Maximum PX4 pose/attitude timestamp difference for a wide-camera map sample (s).')

    parser.add_argument('--simulated-drop-evaluation', action=argparse.BooleanOptionalAction, default=True,
                        help='Evaluate each release as a virtual ballistic payload in the simulator.')
    parser.add_argument('--sim-drop-hit-radius', type=float, default=0.5,
                        help='Maximum predicted impact error for a successful precise drop (m).')
    parser.add_argument('--sim-drop-gravity', type=float, default=9.80665,
                        help='Down-positive gravity for virtual payload propagation (m/s²).')
    parser.add_argument('--sim-drop-wind-north', type=float, default=0.0,
                        help='Constant northward virtual payload drift velocity (m/s).')
    parser.add_argument('--sim-drop-wind-east', type=float, default=0.0,
                        help='Constant eastward virtual payload drift velocity (m/s).')
    parser.add_argument('--sim-drop-ground-z', type=float, default=None,
                        help='Optional fixed local-NED impact-plane down coordinate; omit to use latest depth target.')
    parser.add_argument('--sim-drop-target-max-age', type=float, default=1.2,
                        help='Maximum age of the depth target measurement accepted at release (s).')
    parser.add_argument('--drop-eval-log-dir', type=str, default='~/flylogs',
                        help='Directory for CSV records of virtual payload releases.')
    
    
     # --- 定时器参数 ---
    parser.add_argument('--timer-period', type=float, default=0.03,
                        help='定时器周期 (秒), 这也决定了PID控制中的 dt。默认: 0.03s (约33Hz).')
    parser.add_argument('--vision-timer-period', type=float, default=0.1,
                        help='定时器周期 (秒), 默认: 0.1s (10Hz).')
    parser.add_argument('--image-timeout', type=float, default=3.0,
                        help='仿真图像超过该时长未更新时才判定断流并触发安全处置（秒）。')


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
    
    parser.add_argument('--tracking-buffer', type=int, default=25, help='Number of frames for tracking history.')

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
    parser.add_argument('--alignment-altitude-threshold', type=float, default=0.2,
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
    parser.add_argument('--recon-search-timeout', type=float, default=7.0,
                        help='侦察阶段视觉搜索的持续时间（秒）。')
    parser.add_argument('--recon-hover-time', type=float, default=3.0,
                        help='到达每个侦察圆筒上方后的悬停侦察时间（秒）。')
    parser.add_argument('--recon-nav-threshold', type=float, default=0.5,
                        help='判断无人机到达侦察点的误差阈值（米）。')
    
    parser.add_argument('--recon-forward-distance', type=float, default=7.0,
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

    if not custom_args.headless and not opencv_gui_available():
        print(
            "警告：当前WSL/远程桌面无法使用OpenCV GUI，自动切换到 --headless；"
            "视觉推理和录像仍会继续。",
            file=sys.stderr,
        )
        custom_args.headless = True

    if (
        not math.isfinite(custom_args.target_timeout_duration)
        or custom_args.target_timeout_duration <= 0.0
    ):
        parser.error('--target-timeout-duration must be positive')
    if (
        not math.isfinite(custom_args.target_confidence_window)
        or custom_args.target_confidence_window <= 0.0
    ):
        parser.error('--target-confidence-window must be positive')
    if (
        not math.isfinite(custom_args.target_anchor_hold_duration)
        or custom_args.target_anchor_hold_duration < 0.0
    ):
        parser.error('--target-anchor-hold-duration must be non-negative')
    if (
        not math.isfinite(custom_args.target_pose_max_skew)
        or custom_args.target_pose_max_skew <= 0.0
    ):
        parser.error('--target-pose-max-skew must be positive')
    if (
        not math.isfinite(custom_args.target_pose_attitude_max_skew)
        or custom_args.target_pose_attitude_max_skew <= 0.0
    ):
        parser.error('--target-pose-attitude-max-skew must be positive')
    if not custom_args.target_observation_frame_id.strip():
        parser.error('--target-observation-frame-id must not be empty')
    if (
        not math.isfinite(custom_args.alignment_altitude_threshold)
        or custom_args.alignment_altitude_threshold <= 0.0
    ):
        parser.error('--alignment-altitude-threshold must be positive')
    for option, value in (
        ('--align-maxstep', custom_args.align_maxstep),
        ('--first-align-threshold', custom_args.first_align_threshold),
        ('--second-align-threshold', custom_args.second_align_threshold),
        ('--first-align-maxtime', custom_args.first_align_maxtime),
        ('--second-align-maxtime', custom_args.second_align_maxtime),
    ):
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f'{option} must be positive')
    for option, value in (
        ('--first-align-time-window', custom_args.first_align_time_window),
        ('--second-align-time-window', custom_args.second_align_time_window),
    ):
        if not math.isfinite(value) or value < 0.0:
            parser.error(f'{option} must be non-negative')
    if not 0.0 < custom_args.widecam_min_ground_ray_down < 1.0:
        parser.error('--widecam-min-ground-ray-down must be in (0, 1)')
    if custom_args.widecam_max_ground_range <= 0.0:
        parser.error('--widecam-max-ground-range must be positive')
    if custom_args.widecam_map_max_speed < 0.0:
        parser.error('--widecam-map-max-speed must be non-negative')
    if custom_args.widecam_map_outlier_floor <= 0.0:
        parser.error('--widecam-map-outlier-floor must be positive')
    if custom_args.widecam_max_pose_attitude_skew <= 0.0:
        parser.error('--widecam-max-pose-attitude-skew must be positive')
    if custom_args.sim_drop_hit_radius <= 0.0:
        parser.error('--sim-drop-hit-radius must be positive')
    if custom_args.sim_drop_gravity <= 0.0:
        parser.error('--sim-drop-gravity must be positive')
    if custom_args.sim_drop_target_max_age <= 0.0:
        parser.error('--sim-drop-target-max-age must be positive')

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
    offboard_control = None
    try:
        offboard_control = OffboardControl(args=custom_args)
        spin_control_node_safely(offboard_control)
    except KeyboardInterrupt:
        print("程序被用户中断 (Ctrl+C)")
    except (ExternalShutdownException, ShutdownException):
        print("ROS上下文已关闭，准备清理控制节点。")
    finally:
        # 确保节点在退出时被正确销毁，从而触发我们的清理逻辑
        if offboard_control is not None:
            offboard_control.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
