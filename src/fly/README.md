# 说明
这是25赛季仿真控制代码的存档，删去了其他无用的代码。<br>
运行的主程序为`sim`文件夹中的`0809_sim_mono.py`
## 这个ros2包的名称为control

```bash
ros2 run control test
```
在`colcon build`之后，在工作空间根目录运行以上代码即可运行`0809_sim_mono.py`<br>
在`setup.py`中，设置的程序入口名称为`test`,你可以自行更改。<br>
程序运行的是视觉模型文件是`models`下的`best_sim.pt`,这是yolov8适用于仿真环境的模型，后续如果有需要可自行训练其他。<br>



>**祝你好运**
