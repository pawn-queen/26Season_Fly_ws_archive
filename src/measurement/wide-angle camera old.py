import cv2
import numpy as np
import time

# --- 配置参数 ---
CHESSBOARD_SIZE = (9, 6)  # 棋盘格内部角点数 (列, 行)
SQUARE_SIZE_MM = 23       # 单个棋盘格的物理尺寸（毫米），用于真实世界尺度，可选
MIN_IMAGES = 10           # 建议的最小图像数量

# 终止标准
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 准备世界坐标系中的3D点 (0,0,0), (1,0,0), ..., (8,5,0)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
# 如果需要真实尺度，请取消下一行的注释
# objp = objp * SQUARE_SIZE_MM

# 存储采集到的点
objpoints = []  # 世界坐标系中的3D点
imgpoints = []  # 图像中的2D点

# --- 初始化摄像头 ---
cap = cv2.VideoCapture(0) # 您可以根据需要更改摄像头索引，如 0, 1, 2
if not cap.isOpened():
    print("错误：无法打开摄像头。")
    exit()

# 获取摄像头分辨率以创建覆盖图
ret, frame = cap.read()
if not ret:
    print("错误：无法从摄像头读取帧。")
    cap.release()
    exit()
h, w = frame.shape[:2]
coverage_map = np.zeros((h, w, 3), dtype=np.uint8)
print("--- 相机标定程序 ---")
print("操作指南:")
print("  - 移动棋盘格，从不同角度、距离和位置进行拍摄。")
print("  - 观察 'Coverage Map' 窗口，尽量让绿色圆点覆盖整个画面。")
print(f"  - 按 's' 键保存当前帧的角点 (已保存: 0 / {MIN_IMAGES})。")
print("  - 按 'c' 键开始标定 (当图像数量足够时)。")
print("  - 按 'q' 键退出程序。")
print("-" * 20)


# --- 主循环：采集图像 ---
is_collecting = True
while is_collecting:
    ret, frame = cap.read()
    if not ret:
        print("无法接收帧，退出...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    
    display_frame = frame.copy()

    # 如果找到角点
    if found:
        # 优化角点位置
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # 在实时画面上绘制角点
        cv2.drawChessboardCorners(display_frame, CHESSBOARD_SIZE, corners2, found)
        # 在覆盖图上绘制角点
        for point in corners2:
            cv2.circle(coverage_map, tuple(point[0].astype(int)), 3, (0, 255, 0), -1)
    
    # --- 在屏幕上显示提示信息 ---
    num_saved = len(objpoints)
    status_text = f"Saved: {num_saved}/{MIN_IMAGES}"
    cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if num_saved >= MIN_IMAGES:
        cv2.putText(display_frame, "Ready! Press 'c' to Calibrate", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(display_frame, "Press 's' to save corners", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(display_frame, "Press 'q' to Quit", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 显示窗口
    cv2.imshow('Live Camera Feed', display_frame)
    cv2.imshow('Coverage Map', coverage_map)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        is_collecting = False
    elif key == ord('s') and found:
        # 保存数据点
        imgpoints.append(corners2)
        objpoints.append(objp)
        print(f"成功保存第 {len(objpoints)} 张图像的角点。")
    elif key == ord('c'):
        if len(objpoints) >= MIN_IMAGES:
            print("\n已满足最低图像数量要求，准备开始标定...")
            is_collecting = False
        else:
            print(f"\n错误：图像数量不足。当前 {len(objpoints)} 张，至少需要 {MIN_IMAGES} 张。")

# --- 释放资源 ---
print("图像采集结束，正在释放摄像头和窗口...")
cap.release()
cv2.destroyAllWindows()
# 额外等待一小段时间确保窗口完全关闭
time.sleep(0.5) 


# --- 开始标定计算 ---
if len(objpoints) >= MIN_IMAGES:
    print(f"\n正在使用 {len(objpoints)} 张图像进行标定。")
    print("这可能需要1-3分钟，请耐心等待...")
    
    try:
        # 这是核心的、耗时的计算步骤
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        if ret:
            print("\n标定成功！")
            print("\n相机内参矩阵 (Camera Matrix):")
            print(mtx)
            print("\n畸变系数 (Distortion Coefficients):")
            print(dist)
            
            # 计算重投影误差
            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error
            
            reprojection_error = mean_error / len(objpoints)
            print(f"\n平均重投影误差 (Re-projection Error): {reprojection_error:.4f}")
            print("（提示：误差值小于 0.5 通常被认为是良好的标定结果）")

            # 保存标定结果
            np.savez('camera_calibration_data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
            print("\n标定数据已成功保存到 'camera_calibration_data.npz' 文件中。")

    except Exception as e:
        print(f"\n标定过程中发生错误: {e}")
        print("这通常是由于图像数据多样性不足导致的。请确保从非常不同的角度（特别是大倾斜角）拍摄棋盘格。")

else:
    print("\n没有进行标定，因为有效的图像数量不足。")

print("\n程序结束。")
