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

## 一键启动仿真栈(额，假如确保不了文件路径就算了，26Season_Fly_ws_archive和PX4-1.17.0-2026_Season均在uav这个文件夹底下)

仓库提供了 `scripts/sim_stack.sh`，用于在一个 `tmux` 会话中启动完整仿真链路：

1. PX4 Gazebo 仿真；
2. Micro XRCE-DDS Agent；
3. `ros_gz_bridge` 桥接；
4. `detect` 视觉检测节点；
5. `control` 飞控任务节点。

更换模型后重新编译：
```bash
cd /home/queen/uav/26Season_Fly_ws_archive
colcon build --packages-select control detect
source install/setup.bash
```

默认启动：
```bash
./scripts/sim_stack.sh start
```

停止整套仿真栈：

```bash
./scripts/sim_stack.sh stop
```

重新进入已启动的会话：

```bash
./scripts/sim_stack.sh attach
```

查看会话状态：

```bash
./scripts/sim_stack.sh status
```

常用参数示例：

```bash
./scripts/sim_stack.sh start --control-headless --detect-show-image false
```

指定检测模型或控制参数时，通过环境变量传入额外参数：

```bash
DETECT_ARGS="-p weights_path:=/abs/path/model.pt" ./scripts/sim_stack.sh restart
CONTROL_ARGS="--recon-search-timeout 12" ./scripts/sim_stack.sh start
```

如果 PX4 或工作空间路径不同，可以覆盖默认路径：

```bash
PX4_DIR=/abs/path/PX4 WS_DIR=/abs/path/26Season_Fly_ws_archive ./scripts/sim_stack.sh start
```
常用窗口切换方法：
Ctrl+b 0    切到第 0 个窗口
Ctrl+b 1    切到第 1 个窗口
Ctrl+b 2    切到第 2 个窗口
Ctrl+b n    下一个窗口
Ctrl+b p    上一个窗口
Ctrl+b w    显示窗口列表，用方向键选择，回车进入
## 环境要求

本工作空间需要配合兼容 PX4 1.17 的 Gazebo 仿真环境使用。

使用前请确保：

1. 已正确安装 ROS 2 Humble；
2. 已完成 PX4 1.17 仿真环境配置；
3. 当前终端已正确加载 ROS 2 和本工作空间环境；
4. `px4_msgs` 与当前使用的 PX4 版本保持一致。
5. 使用一键启动脚本时，需要安装 `tmux`，并确保 `MicroXRCEAgent` 可直接执行，且路径正确。
