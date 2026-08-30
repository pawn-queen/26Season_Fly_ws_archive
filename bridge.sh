#!/bin/bash
set -euo pipefail

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_FILE="$SCRIPT_DIR/bridge_config.yaml"

echo "配置文件路径: $CONFIG_FILE"

# PX4 1.17 使用 Gazebo Harmonic（gz-msgs10/gz-transport13）。系统中原有的
# ros_gz_bridge 是 Garden 版本（gz-msgs9/gz-transport12），它会在图像桥接时
# 输出 "Unknown message type [8]/[9]"，导致 /camera 没有 ROS 图像帧。
# 使用随工作区配置的用户级 Harmonic bridge，且直接执行该二进制，避免 ros2 run
# 按环境中 Garden 包的索引选择错误版本。
HARMONIC_BRIDGE_PREFIX="${ROS_GZ_HARMONIC_PREFIX:-$HOME/.local/ros_gz_harmonic}"
HARMONIC_BRIDGE_BIN="$HARMONIC_BRIDGE_PREFIX/opt/ros/humble/lib/ros_gz_bridge/parameter_bridge"
HARMONIC_BRIDGE_LIB="$HARMONIC_BRIDGE_PREFIX/opt/ros/humble/lib/libros_gz_bridge_lib.so"

if [[ ! -x "$HARMONIC_BRIDGE_BIN" || ! -f "$HARMONIC_BRIDGE_LIB" ]]; then
    echo "错误：未找到 Gazebo Harmonic 的 ros_gz_bridge。" >&2
    echo "请安装 ros-humble-ros-gzharmonic-bridge，或将 ROS_GZ_HARMONIC_PREFIX 指向其安装前缀。" >&2
    exit 1
fi

export LD_LIBRARY_PATH="$HARMONIC_BRIDGE_PREFIX/opt/ros/humble/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "使用 Harmonic ros_gz_bridge: $HARMONIC_BRIDGE_BIN"
exec "$HARMONIC_BRIDGE_BIN" --ros-args -p config_file:="$CONFIG_FILE"
