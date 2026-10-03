# 说明
感谢上赛季的开源
这是26赛季仿真控制代码的存档.<br>
运行的主程序为`sim`文件夹中的`0707.py`
## 这个ros2包的名称为control
```bash
ros2 run control test
```
在`colcon build`之后，在工作空间根目录运行以上代码即可运行`0707.py`<br>
在`setup.py`中，设置的程序入口名称为`test`,你可以自行更改。<br>
程序运行的视觉模型文件是`models`下的`26n_0807_bright_needle.pt`，由 control 和 detect 各自的包安装目录加载；后续如果有需要可自行训练其他模型。<br>

## 同步实机主逻辑后的仿真启动

`sim/0707.py` 与 `0821auto.py` 使用相同的平滑转场和目标锚点策略，仿真继续从 ROS 图像话题 `/camera` 获取画面，检测节点使用 `/camera` 和 `/depth_camera`。启动入口仍为 `ros2 run control test`，无需切换到实机相机或 TensorRT 模型。

修改共享控制模块后，在工作空间重新编译并加载环境：

```bash
cd /home/queen/uav/26Season_Fly_ws_archive
source /opt/ros/humble/setup.bash
colcon build --packages-select control detect --symlink-install
source install/setup.bash
```

本机需要已配置 PX4 1.17、Gazebo Harmonic、`tmux` 和 `MicroXRCEAgent`；`bridge.sh` 使用工作空间配置的 Harmonic bridge。一键脚本默认启动 `make px4_sitl gz_x500_depth`、UDP 8888 Agent、相机桥接、检测节点和控制节点：

```bash
./scripts/sim_stack.sh start --control-headless --detect-show-image false
```

平滑转场默认关闭；显式传入 `true` 后，飞向投放区和侦察区时使用平滑位置设定点：

```bash
CONTROL_ARGS="--enable-smooth-transit true --target-anchor-mode max-confidence" \
  ./scripts/sim_stack.sh start --control-headless --detect-show-image false
```

`--target-anchor-mode` 支持 `max-confidence`（默认，选择有效观测中置信度最高的单个坐标）和 `top25`（取有效观测中置信度最高的 25%，计算各坐标分量的中位数）。两种策略均使用 4 秒滚动窗口、首次锁定置信度 0.8、锚点距离门限 0.6 米，并在新观测中断后最多保留锚点 2.5 秒；更高置信度观测的重锚规则与实机一致。

仿真保留适合当前场景的默认参数：向投放区前进 3 米、投放后向侦察区前进 7 米、起飞高度相对初始位置为 -2.8 米、投放区搜索高度为 -5.5 米、侦察搜索高度为 -5.0 米（NED 向下为正）。首次和二次对准超时分别为 15/10 秒；实机保留 12/8 秒。

侦察搜索的 7 秒默认计时包含爬升过程，到期时使用最新一帧视觉结果；若该帧没有有效目标，程序会结束侦察并返航。本次 SITL 验证通过临时追加 `--recon-search-timeout 15 --recon-search-height -4.5` 覆盖了侦察点导航和悬停流程；这些启动参数不会改变源码默认值。

已有同名仿真会话时，先停止或使用 `restart`；停止整套仿真：

```bash
./scripts/sim_stack.sh stop
```
