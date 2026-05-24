import numpy as np
import math

class MapInterface:
    def __init__(self, cfg, vehicle_params):
        self.cfg = cfg
        self.vp = vehicle_params
        
        # 将地图原点设为 0，使地图的绝对物理范围为 [0, 50] x [0, 50]
        self.origin_x = 0.0
        self.origin_y = 0.0
        
        self.grid_size_x = int(self.cfg.map_size_x / self.cfg.xy_resolution)
        self.grid_size_y = int(self.cfg.map_size_y / self.cfg.xy_resolution)
        
        # 0为自由，1为障碍物
        self.obstacle_grid = np.zeros((self.grid_size_x, self.grid_size_y), dtype=np.int8)

        self.eps= 0.1 # 碰撞检测时的采样偏移量（半个栅格）

    def add_obstacle_circle(self, cx, cy, radius):
        """添加圆形障碍物（对于栅格中心点，取其上下左右距离eps的四个点进行判断）"""
        x_centers = self.origin_x + (np.arange(self.grid_size_x) + 0.5) * self.cfg.xy_resolution
        y_centers = self.origin_y + (np.arange(self.grid_size_y) + 0.5) * self.cfg.xy_resolution
        xv, yv = np.meshgrid(x_centers, y_centers, indexing='ij')
        
        # 判断四个采样点：右、左、上、下分别偏移 eps
        eps=self.eps
        is_in_right = (xv + eps - cx)**2 + (yv - cy)**2 <= radius**2
        is_in_left  = (xv - eps - cx)**2 + (yv - cy)**2 <= radius**2
        is_in_up    = (xv - cx)**2 + (yv + eps - cy)**2 <= radius**2
        is_in_down  = (xv - cx)**2 + (yv - eps - cy)**2 <= radius**2
        
        occupied = is_in_right | is_in_left | is_in_up | is_in_down
        self.obstacle_grid[occupied] = 1

    @staticmethod
    def _points_in_polygon(xv, yv, polygon_vertices):
        """向量化判断坐标矩阵(xv, yv)中的所有点是否在多边形内部（射线法）"""
        inside = np.zeros_like(xv, dtype=bool)
        n = len(polygon_vertices)
        p1x, p1y = polygon_vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon_vertices[i % n]
            cond1 = (p1y > yv) != (p2y > yv)
            if p2y != p1y:
                x_cross = (p2x - p1x) * (yv[cond1] - p1y) / (p2y - p1y) + p1x
                intersect_cond = xv[cond1] < x_cross
                flip = np.zeros_like(xv, dtype=bool)
                flip[cond1] = intersect_cond
                inside ^= flip
            p1x, p1y = p2x, p2y
        return inside

    def add_obstacle_polygon(self, polygon_vertices, eps=0.1):
        """添加多边形障碍物（对于栅格中心点，取其上下左右距离eps的四个点进行判断）"""
        if not polygon_vertices or len(polygon_vertices) < 3:
            return
            
        x_centers = self.origin_x + (np.arange(self.grid_size_x) + 0.5) * self.cfg.xy_resolution
        y_centers = self.origin_y + (np.arange(self.grid_size_y) + 0.5) * self.cfg.xy_resolution
        xv, yv = np.meshgrid(x_centers, y_centers, indexing='ij')
        
        # 判断四个采样点：右、左、上、下分别偏移 eps
        eps=self.eps
        is_in_right = self._points_in_polygon(xv + eps, yv, polygon_vertices)
        is_in_left  = self._points_in_polygon(xv - eps, yv, polygon_vertices)
        is_in_up    = self._points_in_polygon(xv, yv + eps, polygon_vertices)
        is_in_down  = self._points_in_polygon(xv, yv - eps, polygon_vertices)
        
        occupied = is_in_right | is_in_left | is_in_up | is_in_down
        self.obstacle_grid[occupied] = 1

    def is_out_of_bounds(self, x, y):
        """检查单点是否越界"""
        return (x < self.origin_x or x >= self.origin_x + self.cfg.map_size_x or
                y < self.origin_y or y >= self.origin_y + self.cfg.map_size_y)

    def check_collision(self, state):
        """完全基于车辆矩形轮廓的高精度碰撞检测"""
        x, y, yaw = state
        
        # 0. 如果中心点直接越界，提前 return
        if self.is_out_of_bounds(x, y): 
            return True
            
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        
        # 1. 提取车辆四个角点相对于后轴中心的坐标
        corners_local = [
            (self.vp.front_edge_to_center, self.vp.left_edge_to_center),
            (self.vp.front_edge_to_center, -self.vp.right_edge_to_center),
            (-self.vp.rear_edge_to_center, -self.vp.right_edge_to_center),
            (-self.vp.rear_edge_to_center, self.vp.left_edge_to_center)
        ]
        
        # 2. 计算车辆在全局坐标系下的 AABB (Axis-Aligned Bounding Box) 粗筛包围盒
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        for lx, ly in corners_local:
            gx = x + lx * cos_y - ly * sin_y
            gy = y + lx * sin_y + ly * cos_y
            min_x, max_x = min(min_x, gx), max(max_x, gx)
            min_y, max_y = min(min_y, gy), max(max_y, gy)
            
        # 如果车辆整个 AABB 任何一部分越界，视为碰撞
        if self.is_out_of_bounds(min_x, min_y) or self.is_out_of_bounds(max_x, max_y):
             return True

        # 将连续坐标映射到离散栅格的 index 范围
        idx_min = max(0, int((min_x - self.origin_x) / self.cfg.xy_resolution))
        idx_max = min(self.grid_size_x - 1, int((max_x - self.origin_x) / self.cfg.xy_resolution))
        idy_min = max(0, int((min_y - self.origin_y) / self.cfg.xy_resolution))
        idy_max = min(self.grid_size_y - 1, int((max_y - self.origin_y) / self.cfg.xy_resolution))
        
        # 3. 在 AABB 内部遍历检测，进行局部坐标系精筛
        margin = self.cfg.xy_resolution / 2.0 + self.cfg.collision_safety_margin  # 留出半个栅格的容差
        for i in range(idx_min, idx_max + 1):
            for j in range(idy_min, idy_max + 1):
                # 只有当该栅格是障碍物时才进行几何测算
                if self.obstacle_grid[i, j] == 1:
                    cx = self.origin_x + (i + 0.5) * self.cfg.xy_resolution
                    cy = self.origin_y + (j + 0.5) * self.cfg.xy_resolution
                    
                    # 转换障碍物中心点到车辆局部坐标系
                    dx, dy = cx - x, cy - y
                    local_x = dx * cos_y + dy * sin_y
                    local_y = -dx * sin_y + dy * cos_y

                    # 检测障碍物中心点是否落入车辆矩形内部
                    if (-self.vp.rear_edge_to_center - margin <= local_x <= self.vp.front_edge_to_center + margin and
                        -self.vp.right_edge_to_center - margin <= local_y <= self.vp.left_edge_to_center + margin):
                        return True
                        
        return False