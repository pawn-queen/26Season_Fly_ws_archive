import cv2
import os
import re
import subprocess
import signal
import sys

# ---------------- 用户配置 ---------------- #
SAVE_DIR = "/home/wide_angle_train_videos"
FPS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FOURCC = cv2.VideoWriter_fourcc(*'XVID')
CAMERA_NAME_HINT = "imx577"
# ----------------------------------------- #

# 判断是否有图形界面
HAS_DISPLAY = ("DISPLAY" in os.environ) or ("WAYLAND_DISPLAY" in os.environ)


def find_video_device_by_name(name_hint="imx577"):
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True, text=True, check=True
        )
    except FileNotFoundError:
        print("错误: 系统未安装 'v4l2-ctl'，请先安装：sudo apt install v4l-utils")
        return None
    except subprocess.CalledProcessError as e:
        print(f"执行 'v4l2-ctl --list-devices' 出错: {e}")
        return None

    lines = result.stdout.splitlines()
    matched_device_name = False
    for line in lines:
        if name_hint.lower() in line.lower():
            matched_device_name = True
        elif matched_device_name and "\t/dev/video" in line:
            match = re.search(r"(/dev/video\d+)", line)
            if match:
                return match.group(1)
    return None


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    device_path = find_video_device_by_name(CAMERA_NAME_HINT)
    if device_path is None:
        print(f"未找到包含 '{CAMERA_NAME_HINT}' 的摄像头，使用默认 /dev/video0")
        cap = cv2.VideoCapture(0)
    else:
        print(f"找到摄像头: {device_path}")
        cap = cv2.VideoCapture(device_path)

    if not cap.isOpened():
        print("错误: 无法打开摄像头")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    existing_files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".avi")]
    numbers = []
    for f in existing_files:
        name, ext = os.path.splitext(f)
        if name.isdigit():
            numbers.append(int(name))
    index = max(numbers) + 1 if numbers else 1

    def cleanup(sig, frame):
        print("\n录制结束，正在释放资源...")
        cap.release()
        if HAS_DISPLAY:
            cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)

    while True:
        filename = os.path.join(SAVE_DIR, f"{index}.avi")
        print(f"开始录制: {filename}")

        out = cv2.VideoWriter(filename, FOURCC, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取帧，退出。")
                cleanup(None, None)

            out.write(frame)

            if HAS_DISPLAY:
                cv2.imshow("Recording", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # 没有显示环境，只能靠 'q' 无效，按 Ctrl+C 来退出或切换
                pass

        out.release()
        print(f"保存完成: {filename}")
        index += 1


if __name__ == "__main__":
    main()
