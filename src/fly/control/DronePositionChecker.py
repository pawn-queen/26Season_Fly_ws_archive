# control/DronePositionChecker.py

import math
from collections import deque
import time

class DronePositionChecker:
    def __init__(self, logger_func, tolerance=0.17, duration=3.0):
        """
        初始化无人机位置检查器。
        :param logger_func: 日志记录函数。
        :param tolerance: 位置变化容差（米）。
        :param duration: 判断稳定所需的时间（秒）。
        """
        self.logger_func = logger_func
        self.tolerance = tolerance
        self.duration = duration
        self.positions = deque()  # 用于存储最近位置和时间戳
        self.log_counter = 0

    def update_position(self, position):
        """
        更新无人机的当前位置。
        :param position: 位置元组 (x, y, z)。
        """
        current_time = time.time()
        self.positions.append((position, current_time))

        # 移除超出时间窗口的旧数据
        while self.positions and current_time - self.positions[0][1] > self.duration:
            self.positions.popleft()

    def is_stable(self):
        """
        判断无人机当前位置是否稳定。
        :return: 如果稳定返回 True，否则返回 False。
        """
        if not self.positions:
            return False

        time_span = self.positions[-1][1] - self.positions[0][1]
                
        # === 修复 1: 检查时间跨度而不是样本数量 ===
        if not math.isclose(time_span, self.duration, rel_tol=0.1) and time_span < self.duration:
            if self.log_counter % 30 == 0:  # 每秒打印一次日志
                self.logger_func("数据采集中，尚未达到稳定检测所需时间...")
                self.logger_func(f"当前数据时间跨度: {time_span:.6f} / {self.duration} s")
            self.log_counter += 1
            return False

        # === 修复 2: 高效的 O(n) 稳定性计算 ===
        # 初始化最小和最大坐标
        min_pos = list(self.positions[0][0])
        max_pos = list(self.positions[0][0])

        # 一次遍历找到所有坐标的最小和最大值
        for pos_tuple in self.positions:
            pos = pos_tuple[0]
            for i in range(3): # 遍历 x, y, z
                min_pos[i] = min(min_pos[i], pos[i])
                max_pos[i] = max(max_pos[i], pos[i])

        # 计算这个边界框的对角线距离
        max_drift = self._distance(min_pos, max_pos)

        is_currently_stable = max_drift <= self.tolerance
        
        # 为了避免日志刷屏，可以降低打印频率
        if self.log_counter % 30 == 0: # 每秒打印一次日志
            self.logger_func(f"稳定检测中: 最大漂移 {max_drift:.4f} m / 容差 {self.tolerance} m. 状态: {'稳定' if is_currently_stable else '不稳定'}")
        self.log_counter += 1

        return is_currently_stable

    def reset(self):
        """重置检查器状态"""
        self.positions.clear()
        self.log_counter = 0
        self.logger_func("位置检查器已重置。")


    @staticmethod
    def _distance(pos1, pos2):
        """
        计算两点之间的欧几里得距离。
        :param pos1: 点1 (x, y, z)。
        :param pos2: 点2 (x, y, z)。
        :return: 两点之间的距离。
        """
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))