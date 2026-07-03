from collections import deque
import time
import math
class AlignmentChecker:
    def __init__(self, logger_func, threshold=0.15, time_window=2.0, check_frequency=10):
        """
        :param logger_func: 日志记录函数（例如 Node.get_logger().info）
        :param threshold: 误差阈值 (米)
        :param time_window: 时间窗口 (秒)
        :param check_frequency: 误差检查频率 (每秒次数)
        """
        self.logger_func = logger_func
        self.threshold = threshold
        self.time_window = time_window
        self.check_frequency = check_frequency
        self.error_deque = deque(maxlen=int(time_window * check_frequency))  # 固定大小的队列


    def check(self, current_x, current_y, target_x, target_y):
        """
        检查当前位置是否与目标持续对准。
        这个方法是“无状态的”，它只根据历史误差数据返回当前是否满足对准条件。
        :return: 如果在时间窗口内所有误差都小于阈值，则返回 True，否则返回 False。
        """
        # 计算当前位置与目标点的误差
        error = math.sqrt((current_x - target_x)**2 + (current_y - target_y)**2)

        # 将误差记录到队列中
        self.error_deque.append(error)

        # 只有当队列被填满时，才进行判断
        if len(self.error_deque) == self.error_deque.maxlen:
            # 检查队列中的所有误差是否都小于阈值
            if all(e < self.threshold for e in self.error_deque):
                self.logger_func(f"对准条件满足: 连续 {len(self.error_deque)} 次误差小于阈值 {self.threshold} m。")
                return True  # 条件满足

        # 减少不必要的日志输出，可以只在接近对准或调试时打印
        # self.logger_func(f"对准检查中... 当前误差: {error:.3f} m, 队列填充: {len(self.error_deque)}/{self.deque_maxlen}")
        
        return False # 默认返回 False，表示条件不满足

    def reset(self):
        """
        重置检查器状态，清空历史误差数据。
        当一个阶段的对准完成后，开始下一阶段前调用。
        """
        self.error_deque.clear()
        self.logger_func("对准检查器已重置。")
