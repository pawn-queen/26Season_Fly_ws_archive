#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import glob
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


# =========================
# 用户通常只需要改这里
# =========================
CALIBRATION_DIR = "/Users/Pictures/calibration_td"
TEST_IMAGE_PATH = "/Users/Pictures/test.png"   # 不需要测试去畸变时设为 None
OUTPUT_DIR = "./calibration_output"

# 棋盘格“内角点”数量：(columns, rows)
patternSize = (6, 9)

# 不强制填写真实格长。这里令相邻内角点间距 = 1 个任意单位。
# 这不会妨碍 K / D 的估计，只意味着 tvec 的尺度也是任意单位。
OBJECT_POINT_SPACING = 1.0

# 异常帧剔除采用 median + MAD 的稳健统计，而不是固定“1 px”阈值。
OUTLIER_SIGMA = 3.5
MAX_REJECTION_ROUNDS = 5
MIN_KEEP_RATIO = 0.65

# 可选：普通 pinhole 是否启用更高阶 radial rational model。
# 默认 False，保持常见 k1,k2,p1,p2,k3 模型；强广角请重点比较 fisheye。
PINHOLE_RATIONAL_MODEL = False

# 仅用于可选测试图的去畸变预览，不影响标定结果。
PINHOLE_ALPHA = 1.0
FISHEYE_BALANCE = 1.0


@dataclass
class View:
    path: Path
    corners: np.ndarray  # (N, 1, 2), float64
    image_size: tuple[int, int]  # (width, height)


@dataclass
class CalibrationResult:
    model: str
    K: np.ndarray
    D: np.ndarray
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]
    overall_rms: float
    per_view_rms: np.ndarray
    kept_indices: list[int]
    rejected: list[dict]
    rejection_history: list[dict]


def find_calibration_images(directory: str) -> list[Path]:
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(Path(p) for p in glob.glob(str(Path(directory) / pattern)))
    return sorted(set(files))


def make_object_points() -> np.ndarray:
    """生成 Nx3 的平面棋盘格坐标；相邻内角点间距为任意单位 1.0。"""
    cols, rows = patternSize
    objp = np.zeros((cols * rows, 3), dtype=np.float64)
    objp[:, :2] = (
        np.mgrid[0:cols, 0:rows]
        .T.reshape(-1, 2)
        .astype(np.float64)
        * OBJECT_POINT_SPACING
    )
    return objp


def detect_chessboard_views(image_paths: list[Path], show_dir: Path) -> list[View]:
    """使用 findChessboardCornersSB 检测角点，并保存检测可视化。"""
    show_dir.mkdir(parents=True, exist_ok=True)

    flags = (
        cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
    )

    detected: list[View] = []

    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] 无法读取: {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])

        ret, corners = cv2.findChessboardCornersSB(
            gray,
            patternSize,
            flags=flags,
        )

        if not ret or corners is None:
            print(f"[MISS] 未检测到 {patternSize} 内角点: {path.name}")
            continue

        corners = np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2)

        # SB 本身直接返回亚像素角点，不再调用 cornerSubPix。
        vis = img.copy()
        cv2.drawChessboardCorners(vis, patternSize, corners.astype(np.float32), True)
        cv2.imwrite(str(show_dir / f"{path.stem}_corners.jpg"), vis)

        detected.append(View(path=path, corners=corners, image_size=image_size))
        print(f"[ OK ] {path.name}: size={image_size}")

    return detected


def choose_dominant_image_size(views: list[View]) -> tuple[list[View], tuple[int, int]]:
    """
    标定本身不能混用不同像素尺寸的图像。
    不要求用户事先指定尺寸；自动选择成功检测数量最多的尺寸组。
    """
    if not views:
        raise RuntimeError("没有任何成功检测到棋盘格的图片。")

    counts: dict[tuple[int, int], int] = {}
    for v in views:
        counts[v.image_size] = counts.get(v.image_size, 0) + 1

    dominant_size = max(counts, key=counts.get)
    selected = [v for v in views if v.image_size == dominant_size]
    skipped = [v for v in views if v.image_size != dominant_size]

    print("\n[INFO] 成功检测的分辨率分组:")
    for size, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        mark = " <- 使用" if size == dominant_size else ""
        print(f"  {size}: {count} 张{mark}")

    if skipped:
        print(f"[WARN] 有 {len(skipped)} 张已检测图片尺寸不同，为避免混合标定而跳过。")

    return selected, dominant_size


def rms_from_points(observed: np.ndarray, projected: np.ndarray) -> float:
    obs = observed.reshape(-1, 2).astype(np.float64)
    prj = projected.reshape(-1, 2).astype(np.float64)
    sq = np.sum((obs - prj) ** 2, axis=1)
    return float(np.sqrt(np.mean(sq)))


def calibrate_pinhole(
    obj_template: np.ndarray,
    views: list[View],
    indices: list[int],
    image_size: tuple[int, int],
) -> CalibrationResult:
    objpoints = [obj_template.astype(np.float32).copy() for _ in indices]
    imgpoints = [views[i].corners.astype(np.float32).copy() for i in indices]

    flags = 0
    if PINHOLE_RATIONAL_MODEL:
        flags |= cv2.CALIB_RATIONAL_MODEL

    (
        overall_rms,
        K,
        D,
        rvecs,
        tvecs,
        _std_intrinsics,
        _std_extrinsics,
        _opencv_per_view_errors,
    ) = cv2.calibrateCameraExtended(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
        flags=flags,
    )

    # 为 pinhole / fisheye 使用完全一致的 RMS 定义，这里手动再算一次逐图 RMS。
    per_view = []
    for local_i, global_i in enumerate(indices):
        projected, _ = cv2.projectPoints(
            obj_template,
            rvecs[local_i],
            tvecs[local_i],
            K,
            D,
        )
        per_view.append(rms_from_points(views[global_i].corners, projected))

    return CalibrationResult(
        model="pinhole",
        K=np.asarray(K, dtype=np.float64),
        D=np.asarray(D, dtype=np.float64),
        rvecs=list(rvecs),
        tvecs=list(tvecs),
        overall_rms=float(overall_rms),
        per_view_rms=np.asarray(per_view, dtype=np.float64),
        kept_indices=list(indices),
        rejected=[],
        rejection_history=[],
    )


def calibrate_fisheye(
    obj_template: np.ndarray,
    views: list[View],
    indices: list[int],
    image_size: tuple[int, int],
) -> CalibrationResult:
    objpoints = [
        obj_template.reshape(-1, 1, 3).astype(np.float64).copy()
        for _ in indices
    ]
    imgpoints = [
        views[i].corners.reshape(-1, 1, 2).astype(np.float64).copy()
        for i in indices
    ]

    K = np.zeros((3, 3), dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)

    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_CHECK_COND
        | cv2.fisheye.CALIB_FIX_SKEW
    )
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-7,
    )

    overall_rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        image_size,
        K,
        D,
        None,
        None,
        flags,
        criteria,
    )

    per_view = []
    for local_i, global_i in enumerate(indices):
        projected, _ = cv2.fisheye.projectPoints(
            obj_template.reshape(-1, 1, 3).astype(np.float64),
            rvecs[local_i],
            tvecs[local_i],
            K,
            D,
        )
        per_view.append(rms_from_points(views[global_i].corners, projected))

    return CalibrationResult(
        model="fisheye",
        K=np.asarray(K, dtype=np.float64),
        D=np.asarray(D, dtype=np.float64),
        rvecs=list(rvecs),
        tvecs=list(tvecs),
        overall_rms=float(overall_rms),
        per_view_rms=np.asarray(per_view, dtype=np.float64),
        kept_indices=list(indices),
        rejected=[],
        rejection_history=[],
    )


def robust_error_threshold(errors: np.ndarray) -> tuple[float, float, float]:
    """返回 threshold, median, robust_sigma。"""
    errors = np.asarray(errors, dtype=np.float64).ravel()
    median = float(np.median(errors))
    mad = float(np.median(np.abs(errors - median)))
    robust_sigma = 1.4826 * mad

    if robust_sigma > 1e-12:
        threshold = median + OUTLIER_SIGMA * robust_sigma
    else:
        # 当所有误差非常接近时，不用固定 px 阈值；仅容许相对 median 的明显偏离。
        threshold = median * 1.5 + 1e-12

    return float(threshold), median, float(robust_sigma)


def calibrate_with_outlier_rejection(
    model_name: str,
    calibrator: Callable,
    obj_template: np.ndarray,
    views: list[View],
    image_size: tuple[int, int],
) -> CalibrationResult:
    if len(views) < 3:
        raise RuntimeError(f"{model_name}: 有效标定图少于 3 张，无法进行可靠求解。")

    indices = list(range(len(views)))
    min_keep = max(3, int(math.ceil(len(indices) * MIN_KEEP_RATIO)))
    rejected: list[dict] = []
    history: list[dict] = []

    final_result: CalibrationResult | None = None

    for round_idx in range(MAX_REJECTION_ROUNDS + 1):
        result = calibrator(obj_template, views, indices, image_size)
        threshold, median, robust_sigma = robust_error_threshold(result.per_view_rms)

        history.append(
            {
                "round": round_idx,
                "count": len(indices),
                "overall_rms": result.overall_rms,
                "median_view_rms": median,
                "robust_sigma": robust_sigma,
                "threshold": threshold,
            }
        )

        print(
            f"[{model_name}] round={round_idx}, views={len(indices)}, "
            f"overall_RMS={result.overall_rms:.5f}px, "
            f"median={median:.5f}px, threshold={threshold:.5f}px"
        )

        bad_local = np.where(result.per_view_rms > threshold)[0]

        # 没有异常帧 / 达到轮数上限 / 再删会低于保留比例：停止。
        if (
            len(bad_local) == 0
            or round_idx >= MAX_REJECTION_ROUNDS
            or len(indices) <= min_keep
        ):
            final_result = result
            break

        # 为防止一次删掉过多数据，每轮只剔除 RMS 最大的一个异常视图，再重新求解。
        worst_local = int(bad_local[np.argmax(result.per_view_rms[bad_local])])
        worst_global = indices[worst_local]
        worst_error = float(result.per_view_rms[worst_local])

        if len(indices) - 1 < min_keep:
            final_result = result
            break

        rejected.append(
            {
                "round": round_idx,
                "global_index": worst_global,
                "path": str(views[worst_global].path),
                "rms": worst_error,
                "threshold": threshold,
            }
        )
        print(
            f"[{model_name}] reject: {views[worst_global].path.name} "
            f"RMS={worst_error:.5f}px"
        )
        del indices[worst_local]

    if final_result is None:
        final_result = calibrator(obj_template, views, indices, image_size)

    final_result.rejected = rejected
    final_result.rejection_history = history
    final_result.kept_indices = list(indices)
    return final_result


def save_yaml(path: Path, result: CalibrationResult, image_size: tuple[int, int]) -> None:
    """使用 OpenCV FileStorage 输出 OpenCV 原生可读 YAML，不依赖 PyYAML。"""
    w, h = image_size
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f"无法写入 YAML: {path}")

    fs.write("model", result.model)
    fs.write("image_width", int(w))
    fs.write("image_height", int(h))
    fs.write("pattern_columns", int(patternSize[0]))
    fs.write("pattern_rows", int(patternSize[1]))
    fs.write("object_point_spacing", float(OBJECT_POINT_SPACING))
    fs.write("camera_matrix", result.K)
    fs.write("distortion_coefficients", result.D)
    fs.write("overall_rms", float(result.overall_rms))
    fs.write("retained_view_count", int(len(result.kept_indices)))
    fs.write("rejected_view_count", int(len(result.rejected)))
    fs.release()


def save_npz(path: Path, result: CalibrationResult, views: list[View], image_size: tuple[int, int]) -> None:
    used_images = np.asarray(
        [str(views[i].path) for i in result.kept_indices],
        dtype=np.str_,
    )
    rejected_images = np.asarray(
        [item["path"] for item in result.rejected],
        dtype=np.str_,
    )

    np.savez_compressed(
        path,
        model=np.asarray(result.model),
        image_size=np.asarray(image_size, dtype=np.int32),
        pattern_size=np.asarray(patternSize, dtype=np.int32),
        object_point_spacing=np.asarray(OBJECT_POINT_SPACING, dtype=np.float64),
        camera_matrix=result.K,
        distortion_coefficients=result.D,
        overall_rms=np.asarray(result.overall_rms, dtype=np.float64),
        per_view_rms=result.per_view_rms,
        used_images=used_images,
        rejected_images=rejected_images,
    )


def save_per_view_csv(path: Path, result: CalibrationResult, views: list[View]) -> None:
    rejected_by_index = {
        int(item["global_index"]): item
        for item in result.rejected
    }

    final_error_by_global = {
        global_i: float(result.per_view_rms[local_i])
        for local_i, global_i in enumerate(result.kept_indices)
    }

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["status", "image", "final_per_view_rms_px", "rejected_rms_px", "round"])

        for i, view in enumerate(views):
            if i in final_error_by_global:
                writer.writerow([
                    "kept",
                    str(view.path),
                    f"{final_error_by_global[i]:.8f}",
                    "",
                    "",
                ])
            elif i in rejected_by_index:
                item = rejected_by_index[i]
                writer.writerow([
                    "rejected",
                    str(view.path),
                    "",
                    f"{float(item['rms']):.8f}",
                    int(item["round"]),
                ])


def save_history_csv(path: Path, result: CalibrationResult) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "round",
                "count",
                "overall_rms",
                "median_view_rms",
                "robust_sigma",
                "threshold",
            ],
        )
        writer.writeheader()
        writer.writerows(result.rejection_history)


def scaled_K_if_pure_resize(
    K: np.ndarray,
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> np.ndarray | None:
    """仅当长宽缩放比例一致时自动缩放 K；不同宽高比可能涉及 crop，不擅自猜。"""
    sw, sh = src_size
    dw, dh = dst_size
    sx = dw / sw
    sy = dh / sh

    if not np.isclose(sx, sy, rtol=1e-4, atol=1e-6):
        return None

    K2 = K.copy().astype(np.float64)
    K2[0, 0] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


def save_undistort_preview(
    test_image_path: str | None,
    output_dir: Path,
    image_size: tuple[int, int],
    pinhole: CalibrationResult | None,
    fisheye: CalibrationResult | None,
) -> None:
    if not test_image_path:
        return

    img = cv2.imread(str(test_image_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] 测试图无法读取，跳过去畸变预览: {test_image_path}")
        return

    test_size = (img.shape[1], img.shape[0])

    if pinhole is not None:
        K = scaled_K_if_pure_resize(pinhole.K, image_size, test_size)
        if K is None:
            print("[WARN] pinhole 测试图宽高比与标定图不同，可能涉及 crop；不自动缩放 K。")
        else:
            new_K, _roi = cv2.getOptimalNewCameraMatrix(
                K,
                pinhole.D,
                test_size,
                PINHOLE_ALPHA,
                test_size,
            )
            map1, map2 = cv2.initUndistortRectifyMap(
                K,
                pinhole.D,
                None,
                new_K,
                test_size,
                cv2.CV_16SC2,
            )
            dst = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
            cv2.imwrite(str(output_dir / "preview_pinhole.png"), dst)

    if fisheye is not None:
        K = scaled_K_if_pure_resize(fisheye.K, image_size, test_size)
        if K is None:
            print("[WARN] fisheye 测试图宽高比与标定图不同，可能涉及 crop；不自动缩放 K。")
        else:
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K,
                fisheye.D,
                test_size,
                np.eye(3),
                balance=FISHEYE_BALANCE,
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K,
                fisheye.D,
                np.eye(3),
                new_K,
                test_size,
                cv2.CV_16SC2,
            )
            dst = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
            cv2.imwrite(str(output_dir / "preview_fisheye.png"), dst)


def print_result(result: CalibrationResult, views: list[View]) -> None:
    print("\n" + "=" * 72)
    print(f"MODEL: {result.model}")
    print(f"Overall RMS: {result.overall_rms:.6f} px")
    print(f"Kept views:  {len(result.kept_indices)}")
    print(f"Rejected:    {len(result.rejected)}")
    print("K =")
    print(result.K)
    print("D =")
    print(result.D.reshape(-1))

    if result.rejected:
        print("Rejected images:")
        for item in result.rejected:
            print(
                f"  - {Path(item['path']).name}: "
                f"RMS={float(item['rms']):.6f}px, round={item['round']}"
            )

    if len(result.per_view_rms):
        worst_local = int(np.argmax(result.per_view_rms))
        worst_global = result.kept_indices[worst_local]
        print(
            f"Worst retained view: {views[worst_global].path.name}, "
            f"RMS={result.per_view_rms[worst_local]:.6f}px"
        )


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    show_dir = output_dir / "corner_visualization"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = find_calibration_images(CALIBRATION_DIR)
    if not image_paths:
        raise RuntimeError(f"标定目录中没有找到图片: {CALIBRATION_DIR}")

    print(f"[INFO] 找到 {len(image_paths)} 张候选标定图。")
    all_detected = detect_chessboard_views(image_paths, show_dir)
    views, image_size = choose_dominant_image_size(all_detected)

    if len(views) < 3:
        raise RuntimeError(
            f"分辨率 {image_size} 下只有 {len(views)} 张有效棋盘图，数量过少。"
        )

    print(f"\n[INFO] 最终用于候选标定的数据: {len(views)} 张, image_size={image_size}")
    obj_template = make_object_points()

    pinhole_result: CalibrationResult | None = None
    fisheye_result: CalibrationResult | None = None

    # ---------- Pinhole ----------
    try:
        pinhole_result = calibrate_with_outlier_rejection(
            "pinhole",
            calibrate_pinhole,
            obj_template,
            views,
            image_size,
        )
        print_result(pinhole_result, views)
        save_yaml(output_dir / "camera_pinhole.yaml", pinhole_result, image_size)
        save_npz(output_dir / "camera_pinhole.npz", pinhole_result, views, image_size)
        save_per_view_csv(output_dir / "per_view_pinhole.csv", pinhole_result, views)
        save_history_csv(output_dir / "rejection_history_pinhole.csv", pinhole_result)
    except cv2.error as e:
        print(f"[ERROR] pinhole 标定失败:\n{e}")

    # ---------- Fisheye ----------
    try:
        fisheye_result = calibrate_with_outlier_rejection(
            "fisheye",
            calibrate_fisheye,
            obj_template,
            views,
            image_size,
        )
        print_result(fisheye_result, views)
        save_yaml(output_dir / "camera_fisheye.yaml", fisheye_result, image_size)
        save_npz(output_dir / "camera_fisheye.npz", fisheye_result, views, image_size)
        save_per_view_csv(output_dir / "per_view_fisheye.csv", fisheye_result, views)
        save_history_csv(output_dir / "rejection_history_fisheye.csv", fisheye_result)
    except cv2.error as e:
        print(f"[ERROR] fisheye 标定失败:\n{e}")

    save_undistort_preview(
        TEST_IMAGE_PATH,
        output_dir,
        image_size,
        pinhole_result,
        fisheye_result,
    )

    print("\n" + "=" * 72)
    print(f"输出目录: {output_dir.resolve()}")
    print("主要文件:")
    print("  camera_pinhole.yaml / .npz")
    print("  camera_fisheye.yaml / .npz")
    print("  per_view_pinhole.csv / per_view_fisheye.csv")
    print("  rejection_history_*.csv")
    print("  corner_visualization/")
    if TEST_IMAGE_PATH:
        print("  preview_pinhole.png / preview_fisheye.png（模型成功且测试图可处理时）")

    if pinhole_result is not None and fisheye_result is not None:
        print("\n模型 RMS（仅作参考，不建议只凭这一项自动选模型）:")
        print(f"  pinhole : {pinhole_result.overall_rms:.6f} px")
        print(f"  fisheye : {fisheye_result.overall_rms:.6f} px")
        print("还应重点看边缘/四角直线恢复效果，以及实际目标反投影误差。")


if __name__ == "__main__":
    main()
