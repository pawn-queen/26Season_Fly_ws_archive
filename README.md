# 26Season 飞行工作空间

本仓库是 2026 赛季无人机仿真与飞控代码使用的 ROS 2 工作空间。

## 功能包说明

* `detect`：视觉检测节点，以及仿真测试入口。
* `control`：PX4 Offboard 控制节点，以及仿真任务入口。
* `px4_msgs`：与 PX4 1.17 仿真环境匹配的 PX4 ROS 2 消息定义。
* `px4_ros_com`：PX4 与 ROS 2 通信相关的支持功能包。

## 基本使用方法

进入工作空间后，先加载 ROS 2 Humble 环境：

```bash
source /opt/ros/humble/setup.bash
```

然后编译工作空间：

```bash
colcon build --symlink-install
```

编译完成后，加载当前工作空间环境：

```bash
source install/setup.bash
```

## 桥接与测试命令

启动通信桥接脚本：

```bash
./bridge.sh
```

运行视觉检测测试节点：

```bash
ros2 run detect test
```

运行控制测试节点：

```bash
ros2 run control test
```

## 环境要求

本工作空间需要配合兼容 PX4 1.17 的 Gazebo 仿真环境使用。

使用前请确保：

1. 已正确安装 ROS 2 Humble；
2. 已完成 PX4 1.17 仿真环境配置；
3. 当前终端已正确加载 ROS 2 和本工作空间环境；
4. `px4_msgs` 与当前使用的 PX4 版本保持一致。
