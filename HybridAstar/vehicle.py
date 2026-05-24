import math

class VehicleParam:
    def __init__(self):
        # 车辆几何与运动学参数 (单位: 米)
        self.wheel_base = 2.8           # 轴距
        self.front_suspend = 0.96       # 前悬
        self.rear_suspend = 0.929       # 后悬
        self.width = 1.942              # 宽度
        
        # 计算车辆轮廓相对于【后轴中心】的边界
        self.length = self.front_suspend + self.wheel_base + self.rear_suspend
        self.front_edge_to_center = self.front_suspend + self.wheel_base
        self.rear_edge_to_center = self.rear_suspend
        self.left_edge_to_center = self.width / 2.0
        self.right_edge_to_center = self.width / 2.0
        
        # 转向与动力学限制
        self.max_steering_angle = math.radians(35.0)  # 最大前轮转角 (rad)
        # 最小转弯半径： R = L / tan(max_steer)
        self.min_turning_radius = self.wheel_base / math.tan(self.max_steering_angle)