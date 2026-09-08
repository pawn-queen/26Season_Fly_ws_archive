import cv2
import glob
import pickle as pkl
import numpy as np

# 一个用于测试的畸变图像路径
distort_image_path = "/Users/Pictures/test.png"

# 用于存放采集的棋盘格子的数据，尽可能的保证距离不要有太大变化
calibration_images = glob.glob("/Users/Pictures/calibration_td/*.png")
distort_image = cv2.imread(distort_image_path)

# 指定棋盘格子内角点的个数
patternSize = (6, 9)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((patternSize[0] * patternSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = []  # 真实世界的3维坐标
imgpoints = []  # 图像的二维平面

# 开始通过角点来计算图像的二维平面坐标
i = 0
for fname in calibration_images:
    img = cv2.imread(fname)
    print(img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    try:
        ret, corners = cv2.findChessboardCorners(gray, patternSize, None)
    except:
        ret, corners = cv2.findChessboardCorners(gray, patternSize[::-1], None)
    if ret: print(fname)
    # If found, add object points, image points (after refining them)
    if ret:
        i += 1
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # 把找到的角点画出来
        cv2.drawChessboardCorners(img, patternSize, corners2, ret)
        cv2.imshow('img', img)
        # 指定存储路径
        cv2.imwrite('/Users/surui/Pictures/calibration_show/' + str(i) + '.jpg', img)
        # cv2.waitKey(50)
# cv2.destroyAllWindows()

# print(objpoints)
# print(imgpoints)

## 缓存中间结果
pkl.dump([objpoints, imgpoints], open("./calibration.pkl", "wb"))
objpoints, imgpoints = pkl.load(open("./calibration.pkl", "rb"))

print(gray.shape[::-1])
# 利用三维坐标和二维坐标来标定相机
retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# print(cameraMatrix)
# print(distCoeffs)

# 畸变图像矫正
h, w = distort_image.shape[:2]
print(h, w)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))
dst = cv2.undistort(distort_image, cameraMatrix, distCoeffs, None, newcameramtx)
x, y, w, h = roi
print(roi)  # 如果roi都是0的话那么就把这行注释掉，不用ROI进行裁剪
dst = dst[y:y + h, x:x + w]
cv2.imwrite('calibresult.png', dst)

# undistort
mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, None, newcameramtx, (w, h), 5)
dst = cv2.remap(distort_image, mapx, mapy, cv2.INTER_LINEAR)
# crop the image
x, y, w, h = roi  # 如果roi都是0的话那么就把这行注释掉，不用ROI进行裁剪
dst = dst[y:y + h, x:x + w]
cv2.imwrite('calibresult2.png', dst)


# 计算反向投影误差
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
print("total error: {}".format(mean_error / len(objpoints)))