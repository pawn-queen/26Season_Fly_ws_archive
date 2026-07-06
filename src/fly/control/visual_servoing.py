import cv2
from ultralytics import YOLO
from enum import Enum
import math
import numpy as np
import os
import time
from collections import deque, Counter


class VisualServoingController:
    """
    一个独立的视觉伺服控制器类。
    它负责处理图像、运行YOLO模型，并根据其内部状态返回指令。
    ### 新增功能:
    - 接受相机内参以进行位姿估计。
    - 在全局搜索阶段，计算并存储所有目标的FRD坐标。
    """
    def __init__(self, model_path, camera_matrix, dist_coeffs,
                 confidence_threshold=0.5,
                 tracking_buffer_size: int = 30,
                 enable_photo_capture: bool = False,
                 photo_save_path: str = '/tmp/drone_captures',
                 photo_capture_interval: int = 30,
                 enable_video_recording: bool = False,
                 video_save_path: str = '/tmp/drone_videos',
                 video_filename: str = 'output.avi',
                 video_fps: float = 30.0):
        """
        初始化视觉控制器。
        ### 新增参数:
        :param camera_matrix: numpy.array, 相机的内参矩阵 (3x3)。
        :param dist_coeffs: numpy.array, 相机的畸变系数 (1x5)。
        """
        print("视觉控制器：对象已创建，模型待加载。")
        self.model_path = model_path
        self.model = None
        self.is_model_loaded = False

        ### --- 新增: 相机参数 --- ###
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.fx = self.camera_matrix[0, 0]
        self.fy = self.camera_matrix[1, 1]
        self.cx = self.camera_matrix[0, 2]
        self.cy = self.camera_matrix[1, 2]
        print("视觉控制器：相机内参已加载。")
        ### ---------------------- ###

        self.CONFIDENCE_THRESHOLD = confidence_threshold
        

        self.tracking_history = deque(maxlen=tracking_buffer_size)
        self.tracking_buffer_size = tracking_buffer_size
        print(f"视觉控制器：将使用最近 {tracking_buffer_size} 帧进行目标跟踪确认。")
        self.frame_counter = 0

        # ... (其余初始化代码保持不变) ...
        self.enable_photo_capture = enable_photo_capture
        self.photo_save_path = photo_save_path
        self.photo_capture_interval = photo_capture_interval
        if self.enable_photo_capture:
            os.makedirs(self.photo_save_path, exist_ok=True)
            print(f"拍照功能已启用，照片将保存到: {self.photo_save_path}")
        
        self.enable_video_recording = enable_video_recording
        self.video_writer = None
        self.video_fps = video_fps
        self.video_full_path = None
        if self.enable_video_recording:
            os.makedirs(video_save_path, exist_ok=True)
            self.video_full_path = os.path.join(video_save_path, video_filename)
            print(f"视频录制功能已启用，视频将保存为: {self.video_full_path}")


    def load_model(self):
        """加载YOLOv8跟踪模型。"""
        if self.is_model_loaded:
            return True
        try:
            print("视觉控制器：正在加载模型...")
            self.model = YOLO(self.model_path)
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self.model.track(dummy_frame, persist=True, verbose=False)
            self.is_model_loaded = True
            print("视觉控制器：跟踪模型加载并预热成功。")
            return True
        except Exception as e:
            print(f"视觉控制器：加载模型失败！错误: {e}")
            return False

    def reset_for_new_mission(self):
        """重置整个视觉任务，回到最初的全局搜索状态。"""
        print("视觉控制器：任务重置，返回全局搜索。")
        pass


    ### --- 新增: 核心计算方法 --- ###
    def _pixel_to_world_frd(self, pixel_coord_uv, drone_altitude_z):
        """
        将单个像素点坐标，根据无人机高度，转换为无人机机体坐标系(FRD)下的坐标。
        假定相机垂直朝下安装。

        :param pixel_coord_uv: (u, v) 像素坐标 (图像的列, 图像的行)。
        :param drone_altitude_z: 无人机的Z轴高度 (NED坐标系, 负数表示在空中)。
        :return: (x, y) 在无人机FRD坐标系下的坐标 (米)。
        """
        u_pixel, v_pixel = pixel_coord_uv
        
        # 注意输入格式需要是 (1, 1, 2)
        pixel_point_distorted = np.array([[[u_pixel, v_pixel]]], dtype=np.float32)
        # undistortPoints 返回的是归一化坐标
        x_norm, y_norm = cv2.undistortPoints(
            pixel_point_distorted, self.camera_matrix, self.dist_coeffs
        )[0][0]
        
        # 2. 利用相似三角形原理，从归一化平面投影到地面
        H = -drone_altitude_z
        Xc = x_norm * H
        Yc = y_norm * H

        # 3. 将相机坐标转换为无人机FRD坐标
        x_frd = -Yc
        y_frd = Xc

        return x_frd, y_frd

    def process_frame(self, frame, drone_altitude_z: float, max_targets_to_confirm: int=3):
        """处理单帧图像，进行跟踪、确认、命名和建图。"""
        if not self.is_model_loaded:
            return [], frame
        
        # 如果被告知不需要确认任何目标，就直接返回，节省计算资源
        if max_targets_to_confirm <= 0:
            return [], frame
        
        # 帧计数器增加
        self.frame_counter += 1

        # 1. 跟踪并更新历史记录
        results = self.model.track(frame, persist=True, verbose=False)
        current_frame_detections = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            for box, track_id, conf in zip(boxes, ids, confs):
                if conf > self.CONFIDENCE_THRESHOLD:
                    cx = int((box[0] + box[2]) / 2)
                    cy = int((box[1] + box[3]) / 2)
                    current_frame_detections.append({
                        'id': track_id,
                        'center': (cx, cy),
                        'box': box,
                        'conf': float(conf)
                    })
        self.tracking_history.append(current_frame_detections)

        # 2. 分析历史，找出最稳定的ID
        all_ids_in_history = [det['id'] for frame_dets in self.tracking_history for det in frame_dets]
        id_counts = Counter(all_ids_in_history)
        most_common_ids = [item[0] for item in id_counts.most_common(max_targets_to_confirm)]
        
        # 3. 为已确认的稳定ID分配逻辑名称并计算坐标
        # 这个列表将包含所有视觉信息，供主控程序使用
        confirmed_targets_info = []
        if most_common_ids:
            confirmed_detections_this_frame = []
            for det in current_frame_detections:
                if det['id'] in most_common_ids:
                    confirmed_detections_this_frame.append(det)
            
            confirmed_detections_this_frame.sort(key=lambda d: d['center'][0])
            
            # target_names = ["Left", "Middle", "Right"]
            
            for i, det in enumerate(confirmed_detections_this_frame):
                drop_target_names = ["Left", "Middle", "Right"]
                if i < len(drop_target_names):
                    name = drop_target_names[i]
                else:
                    # 为额外的侦察目标生成通用名称
                    name = f"Recon_{i+1}"

                coords_frd = self._pixel_to_world_frd(det['center'], drone_altitude_z)
                confirmed_targets_info.append({
                    'id': det['id'],
                    'name': name, # 使用动态生成的名称
                    'coords_frd': coords_frd,
                    'center_pixel': det['center'],
                    'conf': det['conf']
                })

        # 4. 可视化
        annotated_frame = frame.copy()
        name_map = {tgt['id']: tgt['name'] for tgt in confirmed_targets_info}

        for det in current_frame_detections:
            is_confirmed = det['id'] in most_common_ids
            color = (0, 255, 0) if is_confirmed else (0, 0, 255)
            box = det['box']
            cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            id_text = f"ID:{det['id']} conf:{det['conf']:.2f}"
            if det['id'] in name_map: id_text += f" ({name_map[det['id']]})"
            cv2.putText(annotated_frame, id_text, (box[0], max(box[1] - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        cv2.putText(annotated_frame, f"Confirmed Targets: {len(confirmed_targets_info)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 5. 实现视频和拍照功能
        if self.enable_video_recording:
            if self.video_writer is None:
                height, width = annotated_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(self.video_full_path, fourcc, self.video_fps, (width, height))
            # 修正：将标注后的帧写入视频
            self.video_writer.write(annotated_frame)

        if self.enable_photo_capture and self.frame_counter % self.photo_capture_interval == 0:
            timestamp = time.strftime("%H%M%S")
            photo_filename = f"capture_{timestamp}_frame{self.frame_counter}.jpg"
            photo_path = os.path.join(self.photo_save_path, photo_filename)
            cv2.imwrite(photo_path, annotated_frame)
            print(f"照片已保存: {photo_path}")


        return confirmed_targets_info, annotated_frame

    def cleanup(self):
        """在程序结束时调用，用于释放资源。"""
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"视频文件已成功保存并关闭: {self.video_full_path}")
