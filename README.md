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

### 仿真投水落点判定

`sim/0707.py` 已把两阶段视觉对准与投放结果串成闭环：全局搜索会使用广角相机外参、拍摄时的飞行器位姿和稳健离群值过滤建立 NED 目标地图；每次实际发送舵机投放指令时，节点会以该瞬间的深度相机目标、投放器位置和 PX4 NED 速度计算虚拟载荷的运动学落点。

Gazebo 的 `x500_depth` 没有水体释放/飞溅碰撞传感器，因此该判定是**确定性的弹道评估**，并不模拟流体。结果会发布到 `/simulated_drop_result`，同时写入 `~/flylogs/simulated_drop_eval_<时间>.csv`。其中 `HIT` 表示预测落点到目标中心的水平误差不大于 `hit_radius_m`；`MISS` 与 `NO_FRESH_DEPTH_TARGET` 等状态会保留原因，不能被当作精准命中。

默认命中半径为世界中 1 m 直径桶的 0.5 m。需要固定地面在本地 NED 中的 down 坐标，或调整装载点/风速模型时，可这样启动：

```bash
CONTROL_ARGS="--sim-drop-hit-radius 0.5 --sim-drop-ground-z 0.0 --dropper-zoffset 0.15 --sim-drop-wind-north 0.0 --sim-drop-wind-east 0.0" \
  ./scripts/sim_stack.sh start --control-headless
```

不设置 `--sim-drop-ground-z` 时，评估会使用释放瞬间的深度目标 down 坐标作撞击平面，可避免假设 PX4 本地原点正好等于地面。可用 `--no-simulated-drop-evaluation` 关闭该功能；广角相机安装偏差则通过 `--widecam-*-offset` 和 `--widecam-*-deg` 标定。

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
