import warnings
warnings.simplefilter('ignore', category=FutureWarning)

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from std_msgs.msg import Float32, String
from cv_bridge import CvBridge
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import csv
import os
import time
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

class YOLOv5ROS2(Node):
    def __init__(self): # <<< MODIFIED:  def init -> def __init__ (这是Python类的标准构造函数名)
        super().__init__('yolov5_ros2')

        # --- 参数声明 ---
        pkg_name = 'detect'  # 替换的包名
        
        try:
            pkg_share_dir = get_package_share_directory(pkg_name)
            
            # 假设你的pt文件在包的根目录
            default_weights_path = os.path.join(pkg_share_dir, 'models', 'sim_white_cylinder_detection_prime_anchor.pt')
            
            # 检查文件是否存在，不存在则使用备用路径
            if not os.path.exists(default_weights_path):
                self.get_logger().warn(f"Default weights file not found at {default_weights_path}")
                # 可以设置为空字符串或其他默认值
                default_weights_path = ""
                
        except PackageNotFoundError:
            self.get_logger().error(f"Package {pkg_name} not found")
            default_weights_path = ""

        self.declare_parameter('weights_path', default_weights_path)
        self.declare_parameter('conf_threshold', 0.4)
        self.declare_parameter('color_topic', '/camera')
        self.declare_parameter('depth_topic', '/depth_camera')
        self.declare_parameter('show_image', True)
        self.declare_parameter('record_rgb_video', False)
        self.declare_parameter('video_output_path', '/home/depth_videos')
        self.declare_parameter('roi_scale', 0.5)
        self.declare_parameter('enable_detection_log', True)
        self.declare_parameter('detection_log_dir', '~/flylogs/detection_eval')

        # --- 获取参数 ---
        weights_path = self.get_parameter('weights_path').get_parameter_value().string_value
        self.weights_path = weights_path
        self.model_name = os.path.basename(weights_path) if weights_path else 'unknown_model'
        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
        color_topic = self.get_parameter('color_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.show_image = self.get_parameter('show_image').get_parameter_value().bool_value
        self.record_rgb = self.get_parameter('record_rgb_video').get_parameter_value().bool_value
        self.video_path = self.get_parameter('video_output_path').get_parameter_value().string_value
        self.roi_scale = self.get_parameter('roi_scale').get_parameter_value().double_value
        self.enable_detection_log = self.get_parameter('enable_detection_log').get_parameter_value().bool_value
        self.detection_log_dir = os.path.expanduser(
            self.get_parameter('detection_log_dir').get_parameter_value().string_value
        )
        if not (0.0 < self.roi_scale <= 1.0):
            self.get_logger().warn("roi_scale must be between 0 and 1. Defaulting to 0.5")
            self.roi_scale = 0.5

        qos_profile_target = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.fx = 465.7411193847656
        self.fy = 465.7411766052246
        self.cx = 320.0
        self.cy = 240.0
        self.intrinsics_received = True
        self.current_mission_state = 'UNKNOWN'
        self.current_mission_phase = 'UNKNOWN'

        # --- 发布者 ---
        self.publisher = self.create_publisher(Point, '/target_position', qos_profile_target)
        self.centerHeight_Pub = self.create_publisher(Float32, '/current_height', 10)
        self.mission_state_subscriber = self.create_subscription(
            String,
            '/mission_state',
            self.mission_state_callback,
            10
        )

        # --- 模型加载 ---
        self.model = YOLO(weights_path)
        self.frame_index = 0
        self.detection_log_file = None
        self.detection_log_writer = None
        if self.enable_detection_log:
            os.makedirs(self.detection_log_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_stem = os.path.splitext(self.model_name)[0].replace(' ', '_')
            self.detection_log_path = os.path.join(
                self.detection_log_dir,
                f"detection_eval_{timestamp}_{model_stem}.csv"
            )
            self.detection_log_file = open(self.detection_log_path, 'w', newline='', encoding='utf-8')
            self.detection_log_writer = csv.writer(self.detection_log_file)
            self.detection_log_writer.writerow([
                'timestamp_sec', 'frame_index', 'mission_state', 'mission_phase',
                'model_name', 'weights_path', 'conf_threshold', 'roi_scale', 'detections_in_frame',
                'target_index', 'class_id', 'class_name', 'confidence',
                'x1', 'y1', 'x2', 'y2', 'center_x', 'center_y',
                'in_roi', 'depth_m', 'world_x', 'world_y', 'world_z', 'published'
            ])
            self.get_logger().info(f"Detection evaluation log: {self.detection_log_path}")

        # --- OpenCV 桥接 ---
        self.bridge = CvBridge()

        # --- 消息同步 ---
        color_sub = message_filters.Subscriber(self, Image, color_topic, qos_profile=qos_profile_sensor_data)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic, qos_profile=qos_profile_sensor_data)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=10, slop=0.5
        )
        self.ts.registerCallback(self.synced_callback)

        # --- 视频录制相关初始化 ---
        self.video_writer = None
        self.is_recording = False
        if self.record_rgb:
            if not os.path.exists(self.video_path):
                os.makedirs(self.video_path)
                self.get_logger().info(f"Created video directory: {self.video_path}")

        self.get_logger().info('YOLOv5 ROS 2 Node Initialized!')
        if self.show_image: self.get_logger().info('Debug image display is ENABLED.')
        
        if self.record_rgb:
            self.get_logger().info(f'RGB video recording is ENABLED. Output path: {self.video_path}')
        else:
            self.get_logger().info('RGB video recording is DISABLED.')
        self.get_logger().info(f"Detection ROI is set to the central {self.roi_scale*100}% of the image.")

        # <<< ADDED: 为 2Hz 发布频率限制做准备 >>>
        self.publish_period = 1.0 / 2.0  # 2 Hz 的周期为 0.5 秒
        self.last_publish_time = self.get_clock().now()
        self.get_logger().info(f"Publisher frequency is limited to {1.0/self.publish_period:.1f} Hz.")

    def destroy_node(self):
        self.get_logger().info("Node is shutting down, attempting to clean up...")
        if self.video_writer is not None:
            self.get_logger().info("Releasing video writer...")
            self.video_writer.release()
            self.get_logger().info("Video writer released.")
        else:
            self.get_logger().warn("Video writer was not initialized, no video to save.")
        if self.detection_log_file is not None and not self.detection_log_file.closed:
            self.detection_log_file.close()
            self.get_logger().info("Detection evaluation log closed.")
        cv2.destroyAllWindows()
        super().destroy_node()

    def write_detection_log(self, row):
        if self.detection_log_writer is None:
            return
        self.detection_log_writer.writerow(row)
        self.detection_log_file.flush()

    def mission_state_callback(self, msg):
        state_text = msg.data.strip()
        if '|' in state_text:
            mission_state, mission_phase = state_text.split('|', 1)
            self.current_mission_state = mission_state.strip() or 'UNKNOWN'
            self.current_mission_phase = mission_phase.strip() or 'UNKNOWN'
        else:
            self.current_mission_state = state_text or 'UNKNOWN'
            self.current_mission_phase = state_text or 'UNKNOWN'

    def camera_info_callback(self, msg):
        # ... (此部分代码不变)
        pass

    def synced_callback(self, color_msg, depth_msg):
        # <<< MODIFIED: 检查是否达到了发布频率 >>>
        current_time = self.get_clock().now()
        elapsed_time = (current_time - self.last_publish_time).nanoseconds / 1e9
        if elapsed_time < self.publish_period:
            return  # 如果时间未到，则直接返回，不处理这一帧

        # 如果时间到了，更新上一次发布的时间戳
        self.last_publish_time = current_time

        if not self.intrinsics_received:
            self.get_logger().warn('Waiting for camera intrinsics, skipping frame...', throttle_duration_sec=2)
            return
        
        try:
            color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            self.process_images(color_image, depth_image)
        except Exception as e:
            self.get_logger().error(f"Failed to process synced images: {e}")

    def get_unique_filename(self):
        # ... (此部分代码不变)
        base_name = "rgb_video"
        ext = ".avi"
        i = 1
        while True:
            file_name = f"{base_name}_{i}{ext}"
            full_path = os.path.join(self.video_path, file_name)
            if not os.path.exists(full_path):
                return full_path
            i += 1
        
    def process_images(self, color_image, depth_image):
        # ... (此方法内的所有代码都不需要改变)
        self.frame_index += 1
        frame_time = self.get_clock().now().nanoseconds / 1e9
        if self.record_rgb and not self.is_recording:
            try:
                filename = self.get_unique_filename()
                fps = 15
                h, w, _ = color_image.shape
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (w, h), isColor=True)
                
                if not self.video_writer.isOpened():
                    self.get_logger().error(f"!!! Failed to open video writer for file: {filename}")
                    self.record_rgb = False
                else:
                    self.is_recording = True
                    self.get_logger().info(f"SUCCESS: Started recording RGB video to {filename}")
            except Exception as e:
                self.get_logger().error(f"!!! EXCEPTION while creating video writer: {e}")
                self.record_rgb = False

        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(color_image)
            
        height, width, _ = color_image.shape
        center_y_img, center_x_img = height // 2, width // 2
        depth_center = depth_image[center_y_img, center_x_img] * 0.001
        
        center_height = Float32()
        center_height.data = float(depth_center)
        self.centerHeight_Pub.publish(center_height)
        
        roi_w = int(width * self.roi_scale)
        roi_h = int(height * self.roi_scale)
        roi_x1 = center_x_img - roi_w // 2
        roi_y1 = center_y_img - roi_h // 2
        roi_x2 = roi_x1 + roi_w
        roi_y2 = roi_y1 + roi_h
        
        display_image = color_image.copy()
        cv2.rectangle(display_image, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
        
        results = self.detect_objects(color_image)
        detections_in_frame = len(results)
        if detections_in_frame == 0:
            self.write_detection_log([
                f"{frame_time:.6f}", self.frame_index,
                self.current_mission_state, self.current_mission_phase,
                self.model_name, self.weights_path, self.conf_threshold, self.roi_scale, 0,
                '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', False
            ])
        
        for i, result in enumerate(results):
            x1, y1, x2, y2, conf, cls = result
            center_x_pixel = int((x1 + x2) / 2)
            center_y_pixel = int((y1 + y2) / 2)
            class_name = self.model.names[int(cls)] if hasattr(self.model, "names") else str(int(cls))
            in_roi = roi_x1 <= center_x_pixel <= roi_x2 and roi_y1 <= center_y_pixel <= roi_y2

            if not in_roi:
                self.get_logger().info(f"[Target {i+1}] is outside ROI, skipping.")
                self.write_detection_log([
                    f"{frame_time:.6f}", self.frame_index,
                    self.current_mission_state, self.current_mission_phase,
                    self.model_name, self.weights_path, self.conf_threshold, self.roi_scale, detections_in_frame,
                    i + 1, int(cls), class_name, f"{conf:.6f}",
                    f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}",
                    center_x_pixel, center_y_pixel, False, '', '', '', '', False
                ])
                continue

            cv2.rectangle(display_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(
                display_image,
                label,
                (int(x1), max(int(y1) - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            self.get_logger().info(f"--- [Target {i+1}] (VALID) Debug Info ---")
            self.get_logger().info(f"  Pixel Coords (u,v): ({center_x_pixel}, {center_y_pixel})")

            depth = self.get_robust_depth(depth_image, center_x_pixel, center_y_pixel)
            self.get_logger().info(f"  Calculated Depth (Z): {depth:.3f} meters")

            X = Y = Z = ''
            published = False
            if depth > 0:
                X, Y, Z = self.pixel_to_world(center_x_pixel, center_y_pixel, depth)
                self.get_logger().info(f"  World Coords (X,Y,Z): ({X:.3f}, {Y:.3f}, {Z:.3f})")
                self.process_and_publish(X, Y, Z)
                published = True
            else:
                self.get_logger().warn(f"  [Target {i+1}] Invalid depth (0), skipping.")
            self.write_detection_log([
                f"{frame_time:.6f}", self.frame_index,
                self.current_mission_state, self.current_mission_phase,
                self.model_name, self.weights_path, self.conf_threshold, self.roi_scale, detections_in_frame,
                i + 1, int(cls), class_name, f"{conf:.6f}",
                f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}",
                center_x_pixel, center_y_pixel, True, f"{depth:.6f}",
                f"{X:.6f}" if published else '',
                f"{Y:.6f}" if published else '',
                f"{Z:.6f}" if published else '',
                published
            ])
            self.get_logger().info(f"--------------------------")

        if self.show_image:
            cv2.imshow("Detection with ROI", display_image)
            cv2.waitKey(1)

    @torch.no_grad()
    def detect_objects(self, image):
        # ... (此部分代码不变)
        results = self.model(image,verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf[0])
            cls = float(box.cls[0])
            if conf > self.conf_threshold:
                detections.append([x1, y1, x2, y2, conf, cls])
        return np.array(detections)

    def get_robust_depth(self, depth_image, x, y, size=5):
        # ... (此部分代码不变)
        h, w = depth_image.shape
        x1 = max(0, x - size // 2); x2 = min(w - 1, x + size // 2)
        y1 = max(0, y - size // 2); y2 = min(h - 1, y + size // 2)
        patch = depth_image[y1:y2+1, x1:x2+1]
        valid_depths = patch[np.isfinite(patch) & (patch > 0)]
        if valid_depths.size > 0:
            return float(np.median(valid_depths))* 0.001
        return 0.0

    def pixel_to_world(self, u, v, depth):
        # ... (此部分代码不变)
        X = (u - self.cx) * depth / self.fx
        Y = (v - self.cy) * depth / self.fy
        Z = depth
        return X, Y, Z

    def process_and_publish(self, X, Y, Z):
        # ... (此部分代码不变)
        point_msg = Point(x=float(X), y=float(Y), z=float(Z))
        self.publisher.publish(point_msg)
        self.get_logger().info(f'Published Target: X={point_msg.x:.3f}, Y={point_msg.y:.3f}, Z={point_msg.z:.3f}')

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv5ROS2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt received, shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': # <<< MODIFIED: name -> __name__, main -> '__main__'
    main()
