#!/usr/bin/env python3
"""Use a RealSense camera and an Ultralytics model for rough 3-D measurement.

The depth stream is aligned to the color stream before sampling.  For each
detection, the program takes a robust median from the central part of the
bounding box and deprojects the box center with the aligned stream intrinsics.

Reported coordinates use the aligned color-camera optical frame, in metres:
    X: right, Y: down, Z: forward (optical-axis depth)

This is a calibration/debugging utility, not a precision metrology tool.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = SCRIPT_DIR.parent / "detect" / "models" / "26n_0807_bright_needle.pt"


@dataclass(frozen=True)
class DepthEstimate:
    """Robust depth result for one bounding-box ROI."""

    depth_m: float
    mad_m: float
    valid_pixels: int
    total_pixels: int
    roi: tuple[int, int, int, int]

    @property
    def valid_ratio(self) -> float:
        return self.valid_pixels / self.total_pixels if self.total_pixels else 0.0


@dataclass(frozen=True)
class ObjectMeasurement:
    """One object's approximate position in aligned color optical axes."""

    detection_index: int
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    depth: Optional[DepthEstimate]
    xyz_m: Optional[tuple[float, float, float]]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "RealSense + YOLO 粗略测量工具：输出目标深度及对齐后的彩色相机"
            "光学坐标系 XYZ，"
            "并保存 RGB、16 位毫米深度图和伪彩深度图。"
        )
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"Ultralytics 模型路径（默认：{DEFAULT_WEIGHTS}）",
    )
    parser.add_argument("--serial", default=None, help="指定 RealSense 序列号；默认使用第一台设备")
    parser.add_argument("--width", type=int, default=640, help="彩色和深度流宽度（默认：640）")
    parser.add_argument("--height", type=int, default=480, help="彩色和深度流高度（默认：480）")
    parser.add_argument("--fps", type=int, default=30, help="相机帧率（默认：30）")
    parser.add_argument("--conf", type=float, default=0.50, help="检测置信度阈值（默认：0.50）")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU 阈值（默认：0.45）")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 推理尺寸（默认：640）")
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics 推理设备，例如 0、cpu；默认由 Ultralytics 自动选择",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="只测量指定类别 ID，例如 --classes 0 2；默认测量所有类别",
    )
    parser.add_argument(
        "--roi-scale",
        type=float,
        default=0.35,
        help="检测框中央取深度区域占框宽高的比例（默认：0.35）",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.15,
        help="接受的最小深度，米（默认：0.15）",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=8.0,
        help="接受的最大深度，米（默认：8.0）",
    )
    parser.add_argument(
        "--min-valid-pixels",
        type=int,
        default=8,
        help="中央 ROI 至少需要的有效深度像素数（默认：8）",
    )
    parser.add_argument(
        "--depth-vis-max",
        type=float,
        default=5.0,
        help="深度伪彩图显示上限，米（默认：5.0）",
    )
    parser.add_argument(
        "--no-depth-filter",
        action="store_true",
        help="禁用 RealSense 空间和时间深度滤波",
    )
    parser.add_argument(
        "--hole-filling",
        action="store_true",
        help="启用孔洞填充；边缘处可能产生偏差，默认关闭",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "realsense_measurements",
        help="调试输出根目录（默认：~/realsense_measurements）",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=30,
        help="每 N 帧保存一组调试图；0 表示仅按 s 手动保存（默认：30）",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=15,
        help="每 N 帧向终端打印测量结果（默认：15）",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=15,
        help="启动后跳过的自动曝光预热帧数（默认：15）",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="不打开 OpenCV 窗口，适合无桌面环境",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="处理指定帧数后退出；0 表示一直运行",
    )
    return parser


def validate_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("--width、--height 和 --fps 必须为正数")
    if args.imgsz <= 0:
        parser.error("--imgsz 必须为正整数")
    finite_values = (
        args.conf,
        args.iou,
        args.roi_scale,
        args.min_depth,
        args.max_depth,
        args.depth_vis_max,
    )
    if not all(math.isfinite(value) for value in finite_values):
        parser.error("置信度、ROI 和深度相关参数必须是有限数值")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.iou <= 1.0:
        parser.error("--conf 和 --iou 必须在 [0, 1] 范围内")
    if not 0.0 < args.roi_scale <= 1.0:
        parser.error("--roi-scale 必须在 (0, 1] 范围内")
    if args.min_depth < 0.0 or args.max_depth <= args.min_depth:
        parser.error("深度范围无效：要求 0 <= --min-depth < --max-depth")
    if args.depth_vis_max <= 0.0:
        parser.error("--depth-vis-max 必须为正数")
    if args.min_valid_pixels <= 0:
        parser.error("--min-valid-pixels 必须为正整数")
    if args.save_every < 0 or args.print_every < 0:
        parser.error("--save-every 和 --print-every 不能为负数")
    if args.warmup_frames < 0 or args.max_frames < 0:
        parser.error("--warmup-frames 和 --max-frames 不能为负数")


def load_runtime_dependencies() -> tuple[Any, Any, Any]:
    """Import hardware/vision dependencies after CLI parsing for useful --help."""

    required = {
        "cv2": "opencv-python",
        "pyrealsense2": "pyrealsense2（或系统安装的 librealsense Python binding）",
        "ultralytics": "ultralytics",
    }
    modules: dict[str, Any] = {}
    missing: list[str] = []
    for module_name, install_name in required.items():
        try:
            modules[module_name] = importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError) as exc:
            missing.append(f"{install_name}: {exc}")

    if missing:
        details = "\n  - ".join(missing)
        raise RuntimeError(
            "缺少运行依赖：\n  - " + details + "\n"
            "请先安装相应依赖；RealSense 还需要 librealsense 驱动和设备访问权限。"
        )

    return modules["cv2"], modules["pyrealsense2"], modules["ultralytics"].YOLO


def central_roi_from_bbox(
    bbox: Sequence[float],
    image_shape: Sequence[int],
    roi_scale: float,
) -> tuple[int, int, int, int]:
    """Clip a bbox and return its central ROI as x1, y1, x2, y2 (exclusive)."""

    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must contain positive height and width")
    if not 0.0 < roi_scale <= 1.0:
        raise ValueError("roi_scale must be in (0, 1]")

    raw_x1, raw_y1, raw_x2, raw_y2 = (float(value) for value in bbox)
    left, right = sorted((raw_x1, raw_x2))
    top, bottom = sorted((raw_y1, raw_y2))
    left = min(max(left, 0.0), float(width))
    right = min(max(right, 0.0), float(width))
    top = min(max(top, 0.0), float(height))
    bottom = min(max(bottom, 0.0), float(height))

    center_x = (left + right) * 0.5
    center_y = (top + bottom) * 0.5
    roi_width = max((right - left) * roi_scale, 1.0)
    roi_height = max((bottom - top) * roi_scale, 1.0)

    x1 = max(0, int(math.floor(center_x - roi_width * 0.5)))
    y1 = max(0, int(math.floor(center_y - roi_height * 0.5)))
    x2 = min(width, int(math.ceil(center_x + roi_width * 0.5)))
    y2 = min(height, int(math.ceil(center_y + roi_height * 0.5)))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def robust_depth_from_bbox(
    depth_m: np.ndarray,
    bbox: Sequence[float],
    roi_scale: float,
    min_depth_m: float,
    max_depth_m: float,
    min_valid_pixels: int,
) -> Optional[DepthEstimate]:
    """Estimate axial depth from valid pixels in a detection's central ROI."""

    if depth_m.ndim != 2:
        raise ValueError("depth_m must be a 2-D array")
    roi = central_roi_from_bbox(bbox, depth_m.shape, roi_scale)
    x1, y1, x2, y2 = roi
    patch = depth_m[y1:y2, x1:x2]
    valid_mask = (
        np.isfinite(patch)
        & (patch >= min_depth_m)
        & (patch <= max_depth_m)
    )
    values = patch[valid_mask].astype(np.float64, copy=False)
    if values.size < min_valid_pixels:
        return None

    initial_median = float(np.median(values))
    initial_mad = float(np.median(np.abs(values - initial_median)))

    # Reject obvious background/foreground outliers while retaining at least
    # a 15 mm band for sensors whose quantisation makes MAD exactly zero.
    inlier_limit_m = max(3.0 * 1.4826 * initial_mad, 0.015)
    inliers = values[np.abs(values - initial_median) <= inlier_limit_m]
    if inliers.size >= min_valid_pixels:
        values = inliers

    median_m = float(np.median(values))
    mad_m = float(np.median(np.abs(values - median_m)))
    return DepthEstimate(
        depth_m=median_m,
        mad_m=mad_m,
        valid_pixels=int(values.size),
        total_pixels=int(patch.size),
        roi=roi,
    )


def deproject_pixel(
    intrinsics: Any,
    pixel: tuple[int, int],
    depth_m: float,
    rs_module: Any,
) -> tuple[float, float, float]:
    """Deproject one pixel into the aligned color-camera optical frame."""

    point = rs_module.rs2_deproject_pixel_to_point(
        intrinsics,
        [float(pixel[0]), float(pixel[1])],
        float(depth_m),
    )
    return float(point[0]), float(point[1]), float(point[2])


def class_name_for(model: Any, class_id: int) -> str:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return str(names.get(class_id, f"class_{class_id}"))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return f"class_{class_id}"


def measurements_from_result(
    result: Any,
    model: Any,
    depth_m: np.ndarray,
    intrinsics: Any,
    args: argparse.Namespace,
    rs_module: Any,
) -> list[ObjectMeasurement]:
    measurements: list[ObjectMeasurement] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return measurements

    for detection_index, box in enumerate(boxes):
        confidence = float(box.conf[0].item())
        class_id = int(box.cls[0].item())
        xyxy = box.xyxy[0].detach().cpu().tolist()
        x1, y1, x2, y2 = (int(round(float(value))) for value in xyxy)
        height, width = depth_m.shape
        center_u = min(max(int(round((x1 + x2) * 0.5)), 0), width - 1)
        center_v = min(max(int(round((y1 + y2) * 0.5)), 0), height - 1)
        depth = robust_depth_from_bbox(
            depth_m,
            (x1, y1, x2, y2),
            args.roi_scale,
            args.min_depth,
            args.max_depth,
            args.min_valid_pixels,
        )
        xyz_m = None
        if depth is not None:
            xyz_m = deproject_pixel(
                intrinsics,
                (center_u, center_v),
                depth.depth_m,
                rs_module,
            )
        measurements.append(
            ObjectMeasurement(
                detection_index=detection_index,
                class_id=class_id,
                class_name=class_name_for(model, class_id),
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                center=(center_u, center_v),
                depth=depth,
                xyz_m=xyz_m,
            )
        )
    return measurements


def depth_colormap(depth_m: np.ndarray, max_depth_m: float, cv2_module: Any) -> np.ndarray:
    """Create a fixed-scale depth colormap; invalid pixels remain black."""

    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    normalised = np.zeros(depth_m.shape, dtype=np.uint8)
    normalised[valid] = np.clip(
        depth_m[valid] / max_depth_m * 255.0,
        0.0,
        255.0,
    ).astype(np.uint8)
    # Invert the scale so nearby objects are warm and distant objects are cool.
    colour_input = np.where(valid, 255 - normalised, 0).astype(np.uint8)
    coloured = cv2_module.applyColorMap(colour_input, cv2_module.COLORMAP_TURBO)
    coloured[~valid] = (0, 0, 0)
    cv2_module.putText(
        coloured,
        f"Depth: 0-{max_depth_m:.1f} m (near=warm)",
        (10, 24),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2_module.LINE_AA,
    )
    return coloured


def draw_measurements(
    image: np.ndarray,
    measurements: Sequence[ObjectMeasurement],
    cv2_module: Any,
) -> np.ndarray:
    annotated = image.copy()
    for measurement in measurements:
        x1, y1, x2, y2 = measurement.bbox
        center_u, center_v = measurement.center
        valid = measurement.xyz_m is not None and measurement.depth is not None
        colour = (0, 220, 0) if valid else (0, 0, 255)
        cv2_module.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
        cv2_module.drawMarker(
            annotated,
            (center_u, center_v),
            colour,
            cv2_module.MARKER_CROSS,
            14,
            2,
        )

        if valid:
            assert measurement.xyz_m is not None
            assert measurement.depth is not None
            x_m, y_m, z_m = measurement.xyz_m
            label_top = (
                f"{measurement.class_name} {measurement.confidence:.2f} "
                f"Z={z_m:.3f}m"
            )
            label_bottom = (
                f"XYZ=({x_m:+.3f},{y_m:+.3f},{z_m:+.3f}) "
                f"valid={measurement.depth.valid_ratio:.0%}"
            )
            rx1, ry1, rx2, ry2 = measurement.depth.roi
            cv2_module.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)
        else:
            label_top = f"{measurement.class_name} {measurement.confidence:.2f} NO DEPTH"
            label_bottom = ""

        text_y = max(y1 - 9, 22)
        cv2_module.putText(
            annotated,
            label_top,
            (max(x1, 0), text_y),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.52,
            colour,
            2,
            cv2_module.LINE_AA,
        )
        if label_bottom:
            cv2_module.putText(
                annotated,
                label_bottom,
                (max(x1, 0), min(max(y2 + 19, 22), annotated.shape[0] - 5)),
                cv2_module.FONT_HERSHEY_SIMPLEX,
                0.45,
                colour,
                1,
                cv2_module.LINE_AA,
            )
    return annotated


def draw_depth_measurements(
    depth_visualisation: np.ndarray,
    measurements: Sequence[ObjectMeasurement],
    cv2_module: Any,
) -> np.ndarray:
    output = depth_visualisation.copy()
    for measurement in measurements:
        colour = (0, 255, 0) if measurement.depth is not None else (0, 0, 255)
        x1, y1, x2, y2 = measurement.bbox
        cv2_module.rectangle(output, (x1, y1), (x2, y2), colour, 2)
        if measurement.depth is not None:
            rx1, ry1, rx2, ry2 = measurement.depth.roi
            cv2_module.rectangle(output, (rx1, ry1), (rx2, ry2), (255, 255, 255), 1)
    return output


def make_side_by_side(
    annotated: np.ndarray,
    depth_visualisation: np.ndarray,
    cv2_module: Any,
) -> np.ndarray:
    if depth_visualisation.shape[:2] != annotated.shape[:2]:
        depth_visualisation = cv2_module.resize(
            depth_visualisation,
            (annotated.shape[1], annotated.shape[0]),
            interpolation=cv2_module.INTER_NEAREST,
        )
    return np.hstack((annotated, depth_visualisation))


def intrinsics_as_dict(intrinsics: Any) -> dict[str, Any]:
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "model": str(intrinsics.model),
        "coeffs": [float(value) for value in intrinsics.coeffs],
    }


def device_as_dict(device: Any, rs_module: Any) -> dict[str, str]:
    info: dict[str, str] = {}
    for key, enum_value in (
        ("name", "name"),
        ("serial", "serial_number"),
        ("firmware", "firmware_version"),
        ("product_line", "product_line"),
    ):
        try:
            camera_info = getattr(rs_module.camera_info, enum_value)
            info[key] = str(device.get_info(camera_info))
        except (AttributeError, RuntimeError):
            info[key] = "unknown"
    return info


def create_session_directory(output_root: Path) -> Path:
    session_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session_dir = output_root.expanduser().resolve() / session_name
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def open_measurement_csv(session_dir: Path) -> tuple[Any, csv.DictWriter]:
    fieldnames = [
        "wall_time_iso",
        "sensor_timestamp_ms",
        "frame_number",
        "detection_index",
        "class_id",
        "class_name",
        "confidence",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "center_u",
        "center_v",
        "depth_m",
        "depth_mad_m",
        "color_camera_x_m",
        "color_camera_y_m",
        "color_camera_z_m",
        "valid_pixels",
        "roi_pixels",
        "valid_ratio",
    ]
    handle = (session_dir / "measurements.csv").open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    handle.flush()
    return handle, writer


def write_measurements(
    writer: csv.DictWriter,
    csv_handle: Any,
    frame_number: int,
    sensor_timestamp_ms: float,
    measurements: Sequence[ObjectMeasurement],
) -> None:
    now_iso = datetime.now().astimezone().isoformat(timespec="milliseconds")
    for measurement in measurements:
        x1, y1, x2, y2 = measurement.bbox
        center_u, center_v = measurement.center
        depth = measurement.depth
        xyz = measurement.xyz_m
        writer.writerow(
            {
                "wall_time_iso": now_iso,
                "sensor_timestamp_ms": f"{sensor_timestamp_ms:.3f}",
                "frame_number": frame_number,
                "detection_index": measurement.detection_index,
                "class_id": measurement.class_id,
                "class_name": measurement.class_name,
                "confidence": f"{measurement.confidence:.6f}",
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "center_u": center_u,
                "center_v": center_v,
                "depth_m": "" if depth is None else f"{depth.depth_m:.6f}",
                "depth_mad_m": "" if depth is None else f"{depth.mad_m:.6f}",
                "color_camera_x_m": "" if xyz is None else f"{xyz[0]:.6f}",
                "color_camera_y_m": "" if xyz is None else f"{xyz[1]:.6f}",
                "color_camera_z_m": "" if xyz is None else f"{xyz[2]:.6f}",
                "valid_pixels": "" if depth is None else depth.valid_pixels,
                "roi_pixels": "" if depth is None else depth.total_pixels,
                "valid_ratio": "" if depth is None else f"{depth.valid_ratio:.6f}",
            }
        )
    csv_handle.flush()


def save_debug_bundle(
    session_dir: Path,
    frame_number: int,
    color_image: np.ndarray,
    annotated: np.ndarray,
    depth_m: np.ndarray,
    depth_visualisation: np.ndarray,
    combined: np.ndarray,
    cv2_module: Any,
) -> None:
    stem = f"frame_{frame_number:06d}"
    depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    depth_mm[valid] = np.clip(
        np.rint(depth_m[valid] * 1000.0),
        0.0,
        float(np.iinfo(np.uint16).max),
    ).astype(np.uint16)

    outputs = {
        session_dir / f"{stem}_color.png": color_image,
        session_dir / f"{stem}_annotated.png": annotated,
        session_dir / f"{stem}_depth_mm.png": depth_mm,
        session_dir / f"{stem}_depth_color.png": depth_visualisation,
        session_dir / f"{stem}_combined.jpg": combined,
    }
    failed: list[str] = []
    for path, image in outputs.items():
        if not cv2_module.imwrite(str(path), image):
            failed.append(str(path))
    if failed:
        raise OSError("以下调试图保存失败：" + ", ".join(failed))


def print_measurements(frame_number: int, measurements: Sequence[ObjectMeasurement]) -> None:
    if not measurements:
        print(f"[frame {frame_number}] 未检测到目标")
        return
    for measurement in measurements:
        if measurement.xyz_m is None or measurement.depth is None:
            print(
                f"[frame {frame_number}] #{measurement.detection_index} "
                f"{measurement.class_name} conf={measurement.confidence:.2f}: 无有效深度"
            )
            continue
        x_m, y_m, z_m = measurement.xyz_m
        print(
            f"[frame {frame_number}] #{measurement.detection_index} "
            f"{measurement.class_name} conf={measurement.confidence:.2f} "
            f"pixel={measurement.center} depth={measurement.depth.depth_m:.3f} m "
            f"XYZ=({x_m:+.3f}, {y_m:+.3f}, {z_m:+.3f}) m "
            f"valid={measurement.depth.valid_pixels}/{measurement.depth.total_pixels}"
        )


def start_realsense(args: argparse.Namespace, rs_module: Any) -> tuple[Any, Any, float]:
    context = rs_module.context()
    devices = context.query_devices()
    if len(devices) == 0:
        raise RuntimeError("未发现 RealSense 设备；请检查 USB3 连接、librealsense 和设备权限")

    if args.serial is not None:
        serials = [
            device.get_info(rs_module.camera_info.serial_number)
            for device in devices
        ]
        if args.serial not in serials:
            raise RuntimeError(
                f"找不到序列号 {args.serial!r}；当前设备：{', '.join(serials)}"
            )

    pipeline = rs_module.pipeline(context)
    config = rs_module.config()
    if args.serial is not None:
        config.enable_device(args.serial)
    config.enable_stream(
        rs_module.stream.depth,
        args.width,
        args.height,
        rs_module.format.z16,
        args.fps,
    )
    config.enable_stream(
        rs_module.stream.color,
        args.width,
        args.height,
        rs_module.format.bgr8,
        args.fps,
    )
    try:
        profile = pipeline.start(config)
    except RuntimeError as exc:
        raise RuntimeError(
            "RealSense 启动失败。请求的彩色/深度分辨率或帧率可能不受支持："
            f"{args.width}x{args.height}@{args.fps}。原始错误：{exc}"
        ) from exc

    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        if not math.isfinite(depth_scale) or depth_scale <= 0.0:
            raise RuntimeError(f"RealSense 返回了无效的 depth scale：{depth_scale}")
    except Exception:
        pipeline.stop()
        raise
    return pipeline, profile, depth_scale


def make_depth_filters(args: argparse.Namespace, rs_module: Any) -> list[Any]:
    filters: list[Any] = []
    if not args.no_depth_filter:
        filters.extend((rs_module.spatial_filter(), rs_module.temporal_filter()))
    if args.hole_filling:
        filters.append(rs_module.hole_filling_filter())
    return filters


def apply_depth_filters(depth_frame: Any, filters: Sequence[Any]) -> Any:
    filtered = depth_frame
    for depth_filter in filters:
        filtered = depth_filter.process(filtered)
    return filtered.as_depth_frame()


def model_predict(model: Any, color_image: np.ndarray, args: argparse.Namespace) -> Any:
    predict_args: dict[str, Any] = {
        "source": color_image,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device is not None:
        predict_args["device"] = args.device
    if args.classes is not None:
        predict_args["classes"] = args.classes
    results = model.predict(**predict_args)
    if not results:
        raise RuntimeError("模型没有返回推理结果")
    return results[0]


def save_session_metadata(
    session_dir: Path,
    args: argparse.Namespace,
    profile: Any,
    depth_scale: float,
    intrinsics: Any,
    rs_module: Any,
) -> None:
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "coordinate_frame": {
            "name": "RealSense aligned color-camera optical frame",
            "unit": "metre",
            "x": "right",
            "y": "down",
            "z": "forward (axial depth)",
        },
        "device": device_as_dict(profile.get_device(), rs_module),
        "depth_scale_m_per_unit": depth_scale,
        "aligned_depth_intrinsics": intrinsics_as_dict(intrinsics),
        "settings": {
            "weights": str(args.weights.resolve()),
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "confidence": args.conf,
            "iou": args.iou,
            "image_size": args.imgsz,
            "classes": args.classes,
            "roi_scale": args.roi_scale,
            "min_depth_m": args.min_depth,
            "max_depth_m": args.max_depth,
            "min_valid_pixels": args.min_valid_pixels,
            "depth_filter": not args.no_depth_filter,
            "hole_filling": args.hole_filling,
        },
    }
    with (session_dir / "session.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def run(args: argparse.Namespace, cv2_module: Any, rs_module: Any, yolo_class: Any) -> int:
    if not args.weights.expanduser().is_file():
        raise FileNotFoundError(f"模型文件不存在：{args.weights.expanduser()}")
    args.weights = args.weights.expanduser().resolve()

    if not args.no_display and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        print("未检测到图形桌面，自动启用 --no-display。", file=sys.stderr)
        args.no_display = True
    if args.no_display and args.save_every == 0:
        print(
            "警告：无显示且 --save-every=0，只会写 measurements.csv，不会保存调试图。",
            file=sys.stderr,
        )

    print(f"正在加载模型：{args.weights}")
    model = yolo_class(str(args.weights))
    print(f"模型类别：{getattr(model, 'names', 'unknown')}")

    pipeline = None
    csv_handle = None
    try:
        pipeline, profile, depth_scale = start_realsense(args, rs_module)
        align = rs_module.align(rs_module.stream.color)
        depth_filters = make_depth_filters(args, rs_module)

        print(f"RealSense 已启动，depth scale={depth_scale:.9f} m/unit")
        print("正在等待自动曝光稳定...")
        aligned_frames = None
        for _ in range(args.warmup_frames):
            aligned_frames = align.process(pipeline.wait_for_frames())

        if aligned_frames is None:
            aligned_frames = align.process(pipeline.wait_for_frames())
        initial_depth = aligned_frames.get_depth_frame()
        initial_color = aligned_frames.get_color_frame()
        if not initial_depth or not initial_color:
            raise RuntimeError("预热后仍未取得对齐的彩色/深度帧")
        intrinsics = initial_depth.profile.as_video_stream_profile().intrinsics

        session_dir = create_session_directory(args.output_dir)
        save_session_metadata(
            session_dir,
            args,
            profile,
            depth_scale,
            intrinsics,
            rs_module,
        )
        csv_handle, csv_writer = open_measurement_csv(session_dir)
        print(f"调试输出目录：{session_dir}")
        print(
            "坐标系：对齐后的彩色相机光学坐标系，X 向右、Y 向下、Z 向前，"
            "单位米；不能直接当作深度相机、机体或 NED 坐标。"
        )
        if not args.no_display:
            print("按 q 或 Esc 退出，按 s 立即保存一组调试图。")

        frame_number = 0
        last_frame_time = time.monotonic()
        smoothed_fps = 0.0
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not aligned_depth_frame:
                print("警告：本帧缺少彩色或对齐深度，已跳过。", file=sys.stderr)
                continue

            filtered_depth_frame = apply_depth_filters(aligned_depth_frame, depth_filters)
            color_image = np.asanyarray(color_frame.get_data())
            depth_units = np.asanyarray(filtered_depth_frame.get_data())
            depth_m = depth_units.astype(np.float32) * depth_scale

            current_intrinsics = (
                filtered_depth_frame.profile.as_video_stream_profile().intrinsics
            )
            result = model_predict(model, color_image, args)
            measurements = measurements_from_result(
                result,
                model,
                depth_m,
                current_intrinsics,
                args,
                rs_module,
            )
            frame_number += 1
            write_measurements(
                csv_writer,
                csv_handle,
                frame_number,
                float(color_frame.get_timestamp()),
                measurements,
            )

            annotated = draw_measurements(color_image, measurements, cv2_module)
            depth_visualisation = depth_colormap(
                depth_m,
                args.depth_vis_max,
                cv2_module,
            )
            depth_visualisation = draw_depth_measurements(
                depth_visualisation,
                measurements,
                cv2_module,
            )
            combined = make_side_by_side(annotated, depth_visualisation, cv2_module)

            now = time.monotonic()
            instant_fps = 1.0 / max(now - last_frame_time, 1e-6)
            smoothed_fps = instant_fps if smoothed_fps == 0.0 else 0.9 * smoothed_fps + 0.1 * instant_fps
            last_frame_time = now
            cv2_module.putText(
                combined,
                f"FPS: {smoothed_fps:.1f} | detections: {len(measurements)}",
                (10, combined.shape[0] - 12),
                cv2_module.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2_module.LINE_AA,
            )

            if args.print_every and frame_number % args.print_every == 0:
                print_measurements(frame_number, measurements)

            save_now = bool(args.save_every and frame_number % args.save_every == 0)
            key = -1
            if not args.no_display:
                cv2_module.imshow("RealSense Object Measurement | RGB + Depth", combined)
                key = cv2_module.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    save_now = True

            if save_now:
                save_debug_bundle(
                    session_dir,
                    frame_number,
                    color_image,
                    annotated,
                    depth_m,
                    depth_visualisation,
                    combined,
                    cv2_module,
                )
                print(f"已保存 frame {frame_number} 调试图")

            if args.max_frames and frame_number >= args.max_frames:
                break

        return 0
    finally:
        if csv_handle is not None:
            try:
                csv_handle.close()
            except Exception as exc:
                print(f"警告：关闭测量 CSV 失败：{exc}", file=sys.stderr)
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception as exc:
                print(f"警告：停止 RealSense 失败：{exc}", file=sys.stderr)
        if not args.no_display:
            try:
                cv2_module.destroyAllWindows()
            except Exception as exc:
                print(f"警告：关闭 OpenCV 窗口失败：{exc}", file=sys.stderr)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    validate_arguments(args, parser)
    try:
        cv2_module, rs_module, yolo_class = load_runtime_dependencies()
        return run(args, cv2_module, rs_module, yolo_class)
    except KeyboardInterrupt:
        print("\n用户中断，正在释放资源。")
        return 130
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
