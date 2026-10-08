import warnings
warnings.simplefilter('ignore', category=FutureWarning)

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import message_filters
from sensor_msgs.msg import CameraInfo, ChannelFloat32, Image, PointCloud
from geometry_msgs.msg import Point, Point32
from std_msgs.msg import Float32, Header
from cv_bridge import CvBridge
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os
import threading
import time
from collections import deque
from detect.target_selection import (
    TARGET_OBSERVATION_FRAME_ID,
    choose_highest_confidence_candidate,
    depth_scale_for_encoding,
    image_timestamps_within_skew,
    source_timestamp_matches_node_clock,
)

class YOLOv5ROS2(Node):
    def __init__(self):
        super().__init__('yolov5_ros2')

        # --- 参数声明 ---
        self.declare_parameter('weights_path', '/home/weights/0728.engine')
        self.declare_parameter('conf_threshold', 0.4)
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')

        self.declare_parameter('show_image', False) 
        self.declare_parameter('publish_debug_image', True)
        # <<< 修改：参数名从 record_depth_video 改为 record_rgb_video，更清晰
        self.declare_parameter('record_rgb_video', False)
        self.declare_parameter('video_output_path', '/home/depth_videos')
        self.declare_parameter('publish_legacy_target_position', False)
        self.declare_parameter('max_rgb_depth_skew', 0.04)
        self.declare_parameter('max_processing_hz', 60.0)
        self.declare_parameter('debug_image_hz', 30.0)
        self.declare_parameter('recording_fps', 15.0)


        # --- 获取参数 ---
        weights_path = self.get_parameter('weights_path').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
        color_topic = self.get_parameter('color_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value

        self.show_image = self.get_parameter('show_image').get_parameter_value().bool_value
        self.publish_debug_image = self.get_parameter(
            'publish_debug_image'
        ).get_parameter_value().bool_value
        # <<< 修改：获取新参数，并使用新变量名 self.record_rgb
        self.record_rgb = self.get_parameter('record_rgb_video').get_parameter_value().bool_value
        self.video_path = self.get_parameter('video_output_path').get_parameter_value().string_value
        self.publish_legacy_target_position = self.get_parameter(
            'publish_legacy_target_position'
        ).get_parameter_value().bool_value
        self.max_rgb_depth_skew = self.get_parameter(
            'max_rgb_depth_skew'
        ).get_parameter_value().double_value
        if (
            not np.isfinite(self.max_rgb_depth_skew)
            or self.max_rgb_depth_skew <= 0.0
        ):
            raise ValueError("max_rgb_depth_skew must be positive")
        self.max_processing_hz = self._positive_rate('max_processing_hz')
        self.debug_image_hz = self._positive_rate('debug_image_hz')
        self.recording_fps = self._positive_rate('recording_fps')

        # ROS reception never waits for inference or video encoding.  Each
        # worker owns at most one pending item; a newer item replaces it.
        self._stop_event = threading.Event()
        self._pair_condition = threading.Condition()
        self._record_condition = threading.Condition()
        self._pending_pair = None
        self._pending_record_frame = None
        self._recent_pair_keys = deque()
        self._seen_pair_keys = set()
        self._last_pair_received_time_ns = None
        self._last_observation_stamp_ns = None
        self._last_rgb_frame_id = None
        self._pair_sequence = 0
        self._record_sequence = 0
        self._inference_thread = None
        self._recording_thread = None
        self._debug_next_at = 0.0
        self._display_lock = threading.Lock()
        self._display_frame = None
        self._display_generation = 0
        self._displayed_generation = -1

        qos_profile_target = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_profile_observation = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        
        qos_profile_intrinsics = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self.intrinsics_received = False # 用于标记是否已收到内参
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            qos_profile_intrinsics  # 使用这个QoS可以确保我们能收到相机节点最后一次发布的"静态"信息
        )

        # --- 发布者 ---
        self.publisher = self.create_publisher(Point, '/target_position', qos_profile_target)
        self.observation_publisher = self.create_publisher(
            PointCloud,
            '/target_observation',
            qos_profile_observation,
        )
        self.centerHeight_Pub = self.create_publisher(Float32, '/current_height', 10)
        self.debug_image_publisher = None
        if self.publish_debug_image:
            self.debug_image_publisher = self.create_publisher(
                Image, '/detect/debug/image', qos_profile_sensor_data
            )

        # --- 模型加载 ---
        self.model = YOLO(weights_path)

        # --- OpenCV 桥接 ---
        self.bridge = CvBridge()

        # --- 消息同步 ---
        color_sub = message_filters.Subscriber(self, Image, color_topic, qos_profile=qos_profile_sensor_data)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic, qos_profile=qos_profile_sensor_data)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub],
            queue_size=5,
            slop=self.max_rgb_depth_skew,
        )
        self.ts.registerCallback(self.synced_callback)

        # --- 视频录制相关初始化 ---
        self.video_writer = None
        self.is_recording = False
        if self.record_rgb: # <<< 修改：检查 self.record_rgb
            # 确保视频保存目录存在
            if not os.path.exists(self.video_path):
                os.makedirs(self.video_path)
                self.get_logger().info(f"Created video directory: {self.video_path}")

        self.get_logger().info('YOLOv5 ROS 2 Node Initialized!')
        if self.show_image: self.get_logger().info('Debug image display is ENABLED.')
        
        # <<< 修改：更新日志信息
        if self.record_rgb:
            self.get_logger().info(f'RGB video recording is ENABLED. Output path: {self.video_path}')
        else:
            self.get_logger().info('RGB video recording is DISABLED.')


    def destroy_node(self):
        """在节点销毁时调用的清理函数，确保资源被释放"""
        self._stop_event.set()
        with self._pair_condition:
            self._pending_pair = None
            self._pair_condition.notify_all()
        with self._record_condition:
            self._pending_record_frame = None
            self._record_condition.notify_all()
        # The existing service stop timeout still bounds a stuck backend.
        # Do not destroy publishers or release a writer underneath its owner.
        for worker in (self._inference_thread, self._recording_thread):
            if (
                worker is not None and worker.ident is not None
                and worker is not threading.current_thread()
            ):
                worker.join()
        context_is_valid = rclpy.ok(context=self.context)
        if context_is_valid:
            self.get_logger().info("Node is shutting down, attempting to clean up...")
        if self.show_image:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        super().destroy_node()

    def _positive_rate(self, name):
        value = self.get_parameter(name).get_parameter_value().double_value
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError('%s must be positive and finite' % name)
        return float(value)

    def start_workers(self):
        """Start one model owner and, if requested, one video writer owner."""
        if self._inference_thread is not None:
            return
        self._inference_thread = threading.Thread(
            target=self._inference_loop, name='detect-inference', daemon=True,
        )
        if self.record_rgb:
            self._recording_thread = threading.Thread(
                target=self._recording_loop, name='detect-recording', daemon=True,
            )
            self._recording_thread.start()
        self._inference_thread.start()

    def _inference_loop(self):
        next_at = 0.0
        while not self._stop_event.is_set():
            with self._pair_condition:
                while not self._stop_event.is_set():
                    if self._pending_pair is None:
                        self._pair_condition.wait()
                        continue
                    delay = next_at - time.monotonic()
                    if delay > 0.0:
                        self._pair_condition.wait(delay)
                        continue
                    pair = self._pending_pair
                    self._pending_pair = None
                    next_at = time.monotonic() + 1.0 / self.max_processing_hz
                    break
                else:
                    return
            if self._stop_event.is_set():
                return
            color_msg, depth_msg, observation_header, intrinsics, _ = pair
            try:
                color_image = self.bridge.imgmsg_to_cv2(
                    color_msg, desired_encoding='bgr8'
                )
                depth_image = self.bridge.imgmsg_to_cv2(
                    depth_msg, desired_encoding='passthrough'
                )
                if not self._stop_event.is_set():
                    self.process_images(
                        color_image, depth_image, depth_msg.encoding,
                        observation_header, color_msg.header,
                        intrinsics=intrinsics,
                    )
            except Exception as exc:
                if not self._stop_event.is_set():
                    self.get_logger().error(
                        f'Failed to process synced images: {exc}',
                        throttle_duration_sec=2,
                    )

    def _offer_recording_frame(self, frame):
        if not self.record_rgb or self._stop_event.is_set():
            return
        with self._record_condition:
            if not self.record_rgb or self._stop_event.is_set():
                return
            self._record_sequence += 1
            # Inputs are read-only; annotations are drawn on a separate copy.
            self._pending_record_frame = (self._record_sequence, frame)
            self._record_condition.notify()

    def _recording_loop(self):
        next_at = 0.0
        try:
            while not self._stop_event.is_set():
                with self._record_condition:
                    while not self._stop_event.is_set():
                        if self._pending_record_frame is None:
                            self._record_condition.wait()
                            continue
                        delay = next_at - time.monotonic()
                        if delay > 0.0:
                            self._record_condition.wait(delay)
                            continue
                        _, frame = self._pending_record_frame
                        self._pending_record_frame = None
                        next_at = time.monotonic() + 1.0 / self.recording_fps
                        break
                    else:
                        return
                if self._stop_event.is_set():
                    return
                if self.video_writer is None:
                    filename = self.get_unique_filename()
                    height, width = frame.shape[:2]
                    self.video_writer = cv2.VideoWriter(
                        filename, cv2.VideoWriter_fourcc(*'XVID'),
                        self.recording_fps, (width, height), isColor=True,
                    )
                    if not self.video_writer.isOpened():
                        raise RuntimeError('Failed to open video writer: ' + filename)
                    self.is_recording = True
                    self.get_logger().info('Started recording RGB video to ' + filename)
                self.video_writer.write(frame)
        except Exception as exc:
            self.record_rgb = False
            self.get_logger().error(
                f'RGB recording stopped after an error: {exc}',
                throttle_duration_sec=2,
            )
        finally:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.is_recording = False
            with self._record_condition:
                self._pending_record_frame = None

    def camera_info_callback(self, msg):
        """
        接收一次相机内参并存储，然后销毁订阅。
        """
        if not self.intrinsics_received:
            self.fx = msg.k[0]  # K[0] is fx
            self.fy = msg.k[4]  # K[4] is fy
            self.cx = msg.k[2]  # K[2] is cx
            self.cy = msg.k[5]  # K[5] is cy
            self.intrinsics_received = True
            self.get_logger().info('Camera intrinsics received successfully!')
            self.get_logger().info(f"  fx: {self.fx}, fy: {self.fy}")
            self.get_logger().info(f"  cx: {self.cx}, cy: {self.cy}")
            # 销毁订阅，因为我们只需要这个信息一次
            self.destroy_subscription(self.camera_info_sub)

    def synced_callback(self, color_msg, depth_msg):
        if self._stop_event.is_set():
            return
        frame_received_at = self.get_clock().now()
        color_stamp_s = (
            float(color_msg.header.stamp.sec)
            + float(color_msg.header.stamp.nanosec) / 1e9
        )
        depth_stamp_s = (
            float(depth_msg.header.stamp.sec)
            + float(depth_msg.header.stamp.nanosec) / 1e9
        )
        if not image_timestamps_within_skew(
            color_stamp_s,
            depth_stamp_s,
            self.max_rgb_depth_skew,
        ):
            self.get_logger().warn(
                "RGB与对齐深度帧时间差超过 %.3fs，丢弃该图像对。"
                % self.max_rgb_depth_skew,
                throttle_duration_sec=2,
            )
            return
        if not self.intrinsics_received:
            self.get_logger().warn('Waiting for camera intrinsics, skipping frame...', throttle_duration_sec=2)
            return
        
        observation_header = self._pose_matching_header(
            color_msg.header, frame_received_at,
        )
        pair_key = (
            color_msg.header.frame_id,
            color_msg.header.stamp.sec, color_msg.header.stamp.nanosec,
        )
        observation_stamp_ns = (
            observation_header.stamp.sec * 1_000_000_000
            + observation_header.stamp.nanosec
        )
        with self._pair_condition:
            if self._stop_event.is_set():
                return
            if (
                self._last_rgb_frame_id != color_msg.header.frame_id
                or (self._last_pair_received_time_ns is not None
                    and frame_received_at.nanoseconds
                    < self._last_pair_received_time_ns)
            ):
                # A new source/ROS clock epoch can establish a new watermark.
                self._recent_pair_keys.clear()
                self._seen_pair_keys.clear()
                self._last_observation_stamp_ns = None
            self._last_rgb_frame_id = color_msg.header.frame_id
            self._last_pair_received_time_ns = frame_received_at.nanoseconds
            if (
                self._last_observation_stamp_ns is not None
                and observation_stamp_ns < self._last_observation_stamp_ns
            ):
                return
            # A zero source stamp has no usable frame identity; retain the
            # reception sequence instead of suppressing all subsequent frames.
            if color_stamp_s > 0.0:
                if pair_key in self._seen_pair_keys:
                    return
                self._seen_pair_keys.add(pair_key)
                self._recent_pair_keys.append(pair_key)
                if len(self._recent_pair_keys) > 128:
                    self._seen_pair_keys.remove(self._recent_pair_keys.popleft())
            self._last_observation_stamp_ns = observation_stamp_ns
            self._pair_sequence += 1
            self._pending_pair = (
                color_msg, depth_msg, observation_header,
                (self.fx, self.fy, self.cx, self.cy), self._pair_sequence,
            )
            self._pair_condition.notify()

    def _pose_matching_header(self, source_header, frame_received_at):
        """Use the source stamp only when it shares this node's clock domain."""
        source_stamp_s = (
            float(source_header.stamp.sec)
            + float(source_header.stamp.nanosec) / 1e9
        )
        received_at_s = frame_received_at.nanoseconds / 1e9
        header = Header(frame_id=TARGET_OBSERVATION_FRAME_ID)
        if source_timestamp_matches_node_clock(source_stamp_s, received_at_s):
            header.stamp = source_header.stamp
        else:
            header.stamp = frame_received_at.to_msg()
            self.get_logger().warn(
                "图像时间戳与节点时钟域不一致，改用该帧进入检测节点的时刻。",
                throttle_duration_sec=5,
            )
        return header

    def get_unique_filename(self):
        """生成一个唯一的视频文件名，避免覆盖"""
        # <<< 修改：更改视频文件名的前缀
        base_name = "rgb_video"
        ext = ".avi"
        i = 1
        while True:
            file_name = f"{base_name}_{i}{ext}"
            full_path = os.path.join(self.video_path, file_name)
            if not os.path.exists(full_path):
                return full_path
            i += 1

    def process_images(
        self,
        color_image,
        depth_image,
        depth_encoding='',
        observation_header=None,
        color_header=None,
        intrinsics=None,
    ):
        if self._stop_event.is_set():
            return
        # Queue a read-only reference; encoding never runs on this thread.
        self._offer_recording_frame(color_image)

        try:
            depth_scale = depth_scale_for_encoding(
                depth_encoding,
                getattr(depth_image.dtype, 'kind', None),
            )
        except ValueError as exc:
            self.get_logger().error(str(exc), throttle_duration_sec=2)
            return

        # --- 获取并发布中心点高度 ---
        height, width = depth_image.shape
        center_y, center_x = height // 2, width // 2
        depth_center = depth_image[center_y, center_x] * depth_scale
        
        center_height = Float32()
        center_height.data = float(depth_center)
        self.centerHeight_Pub.publish(center_height)

        # --- YOLO 目标检测 ---
        results = self.detect_objects(color_image)
        
        candidates = []
        for index, result in enumerate(results):
            x1, y1, x2, y2, conf, cls = result
            if int(cls) != 0:
                continue
            center_x_pixel = int((x1 + x2) / 2)
            center_y_pixel = int((y1 + y2) / 2)
            depth = self.get_robust_depth(
                depth_image,
                center_x_pixel,
                center_y_pixel,
                depth_scale=depth_scale,
            )
            candidates.append({
                'target_index': index,
                'confidence': float(conf),
                'class_id': int(cls),
                'center_x': center_x_pixel,
                'center_y': center_y_pixel,
                'center_distance_sq': (
                    (center_x_pixel - center_x) ** 2
                    + (center_y_pixel - center_y) ** 2
                ),
                'in_roi': True,
                'depth_m': depth,
            })

        selected = choose_highest_confidence_candidate(candidates)

        if self._stop_event.is_set():
            return
        # Publish the unchanged 3D result before auxiliary drawing/packing.
        if selected is not None:
            X, Y, Z = self.pixel_to_world(
                selected['center_x'], selected['center_y'],
                selected['depth_m'], intrinsics=intrinsics,
            )
            self.process_and_publish(
                X, Y, Z, selected['confidence'], observation_header,
            )

        # --- (可选) 调试图像；使用原始 RGB 时间戳与 viewer 缓存帧匹配 ---
        if self.show_image or self.publish_debug_image:
            try:
                should_publish_debug = (
                    self.publish_debug_image
                    and self.debug_image_publisher is not None
                    and color_header is not None
                    and self.debug_image_publisher.get_subscription_count() > 0
                )
                if self.show_image or should_publish_debug:
                    now = time.monotonic()
                    if now < self._debug_next_at:
                        return
                    self._debug_next_at = now + 1.0 / self.debug_image_hz
                    selected_index = None if selected is None else selected['target_index']
                    annotated_image = self.draw_detections(
                        color_image.copy(), results, selected_index
                    )
                    if should_publish_debug and not self._stop_event.is_set():
                        debug_msg = self.bridge.cv2_to_imgmsg(
                            annotated_image, encoding='bgr8'
                        )
                        debug_msg.header = color_header
                        self.debug_image_publisher.publish(debug_msg)
                    if self.show_image:
                        self.show_detections(annotated_image)
                else:
                    # Reopening a viewer need not wait for an old deadline.
                    self._debug_next_at = 0.0
            except Exception as exc:
                self.get_logger().warn(
                    f"Failed to render debug image: {exc}",
                    throttle_duration_sec=2,
                )

    def draw_detections(self, image, detections, selected_index=None):
        for index, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls = det
            class_name = self.model.names[int(cls)] if hasattr(self.model, "names") else str(int(cls))
            is_selected = index == selected_index
            color = (0, 255, 0) if is_selected else (0, 165, 255)
            label = f"{class_name} {conf:.2f}" + (" SELECTED" if is_selected else "")
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(
                image,
                label,
                (int(x1), max(int(y1) - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        return image

    def show_detections(self, image):
        """Hand local GUI output to the main thread, never a model worker."""
        with self._display_lock:
            self._display_frame = image
            self._display_generation += 1

    def display_latest_debug_frame(self):
        with self._display_lock:
            frame = self._display_frame
            generation = self._display_generation
        if frame is None:
            return
        if generation != self._displayed_generation:
            cv2.imshow("Detection", frame)
            self._displayed_generation = generation
        cv2.waitKey(1)

    @torch.no_grad()
    def detect_objects(self, image):
        results = self.model(image,verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf[0])
            cls = float(box.cls[0])
            if conf > self.conf_threshold:
                detections.append([x1, y1, x2, y2, conf, cls])
        return np.array(detections)

    def get_robust_depth(self, depth_image, x, y, size=5, depth_scale=1.0):
        h, w = depth_image.shape
        x1 = max(0, x - size // 2); x2 = min(w - 1, x + size // 2)
        y1 = max(0, y - size // 2); y2 = min(h - 1, y + size // 2)
        patch = depth_image[y1:y2+1, x1:x2+1]
        valid_depths = patch[np.isfinite(patch) & (patch > 0)]
        if valid_depths.size > 0:
            return float(np.median(valid_depths)) * depth_scale
        return 0.0

    def pixel_to_world(self, u, v, depth, intrinsics=None):
        fx, fy, cx, cy = (
            (self.fx, self.fy, self.cx, self.cy)
            if intrinsics is None else intrinsics
        )
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth
        return X, Y, Z

    def process_and_publish(self, X, Y, Z, confidence, observation_header=None):
        observation_msg = PointCloud()
        if observation_header is None:
            observation_msg.header.stamp = self.get_clock().now().to_msg()
            observation_msg.header.frame_id = TARGET_OBSERVATION_FRAME_ID
        else:
            observation_msg.header = observation_header
        observation_msg.points = [Point32(x=float(X), y=float(Y), z=float(Z))]
        observation_msg.channels = [
            ChannelFloat32(name='confidence', values=[float(confidence)])
        ]
        self.observation_publisher.publish(observation_msg)

        if self.publish_legacy_target_position:
            point_msg = Point(x=float(X), y=float(Y), z=float(Z))
            self.publisher.publish(point_msg)
        self.get_logger().info(
            'Published Target: X=%.3f, Y=%.3f, Z=%.3f, confidence=%.3f'
            % (X, Y, Z, confidence),
            throttle_duration_sec=2,
        )

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv5ROS2()
    try:
        node.start_workers()
        if node.show_image:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=1.0 / 30.0)
                node.display_latest_debug_frame()
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt received, shutting down.')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
