"""
使用USB摄像头和YOLOv8的TensorRT引擎进行实时目标检测，并按固定间隔保存照片。
新增功能: 每次运行时，照片会保存在一个以当前时间命名的独立子文件夹中，以防覆盖。
"""
import cv2
import torch
from ultralytics import YOLO
import subprocess
import re
import os
from datetime import datetime # <-- 导入 datetime 用于获取时间戳

# ----------------- 用户配置 ----------------- #
ENGINE_PATH = '/home/weights/best.engine'
CONFIDENCE_THRESHOLD = 0.5
CAMERA_NAME_HINT = "USB"

SAVE_IMAGES = True
# 主保存目录，子文件夹将在此目录下创建
BASE_SAVE_DIR = os.path.expanduser("/home/usb_image_records")
SAVE_INTERVAL_FRAMES = 30
# ------------------------------------------- #

def find_video_device_by_name(name_hint="USB Camera"):
    # ... (此函数保持不变) ...
    try:
        result = subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True, text=True, check=True)
    except FileNotFoundError:
        print("错误: 'v4l2-ctl' 命令未找到。")
        return None
    except subprocess.CalledProcessError as e:
        print(f"执行 'v4l2-ctl --list-devices' 出错: {e}")
        return None
    lines = result.stdout.splitlines()
    matched_device_name = False
    for i, line in enumerate(lines):
        if name_hint in line:
            matched_device_name = True
        elif matched_device_name and "\t/dev/video" in line:
            match = re.search(r"(/dev/video\d+)", line)
            if match:
                return match.group(1)
    return None

def main():
    if not torch.cuda.is_available():
        print("错误：未检测到NVIDIA GPU。TensorRT引擎需要CUDA支持。")
        return
    
    print("正在加载TensorRT引擎...")
    try:
        model = YOLO(ENGINE_PATH)
        print(f"成功加载TensorRT引擎: {ENGINE_PATH}")
    except Exception as e:
        print(f"加载引擎失败: {e}")
        return

    device_path = find_video_device_by_name(CAMERA_NAME_HINT)
    
    if device_path is None:
        print(f"找不到名称包含 '{CAMERA_NAME_HINT}' 的摄像头设备。尝试使用默认设备索引 0。")
        cap = cv2.VideoCapture(0)
    else:
        print(f"找到摄像头设备: {device_path}")
        cap = cv2.VideoCapture(device_path)

    if not cap.isOpened():
        print("错误: 无法打开摄像头。")
        return

    # --- 核心修改: 创建本次运行的唯一保存目录 ---
    frame_counter = 0
    saved_image_counter = 0
    
    # 初始化一个变量来存储本次运行的特定保存路径
    run_save_dir = "" 
    
    if SAVE_IMAGES:
        # 1. 获取当前时间并格式化为字符串，用作文件夹名
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # 2. 结合主目录和时间戳，创建本次运行的完整路径
        run_save_dir = os.path.join(BASE_SAVE_DIR, timestamp)
        
        # 3. 创建这个新目录
        os.makedirs(run_save_dir, exist_ok=True)
        
        print(f"拍照功能已开启，本次运行的照片将保存至: {run_save_dir}")
        print(f"每隔 {SAVE_INTERVAL_FRAMES} 帧保存一张照片。")

    print("摄像头已打开。按 'q' 键退出。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法从摄像头读取帧，程序退出。")
            break

        frame_counter += 1

        # --- 拍照逻辑现在使用 run_save_dir ---
        if SAVE_IMAGES and (frame_counter % SAVE_INTERVAL_FRAMES == 0):
            file_name = f"image_{saved_image_counter:05d}.png"
            # 将照片保存在本次运行的专属文件夹内
            full_path = os.path.join(run_save_dir, file_name)
            
            cv2.imwrite(full_path, frame)
            # 为了简洁，可以在保存大量图片时注释掉这行打印
            # print(f"已保存照片: {full_path}") 
            
            saved_image_counter += 1

        # --- 检测和可视化逻辑保持不变 ---
        results = model(frame, verbose=False)
        result = results[0]

        for box in result.boxes:
            confidence = box.conf[0].item()
            if confidence > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0].item())
                class_name = model.names[class_id] if model.names else f"Class_{class_id}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 TensorRT Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出，资源已释放。")
    if SAVE_IMAGES:
        print(f"本次运行总共在文件夹 '{os.path.basename(run_save_dir)}' 中保存了 {saved_image_counter} 张照片。")

if __name__ == "__main__":
    main()
