import math

class SearchConfig:
    def __init__(self):
        # 地图与分辨率参数
        self.map_size_x = 50.0          
        self.map_size_y = 50.0          
        self.xy_resolution = 0.3        # 空间栅格分辨率
        self.yaw_resolution = math.radians(10.0) # 航向角离散分辨率
        
        # 搜索拓展参数 (State Propagation)
        self.step_arc = 1.0             # 每次拓展的基准弧长
        self.steer_step = math.radians(5.0) # 转角采样步长
        self.check_num = 5              # 弧长拓展段内的碰撞检测插值次数
        
        # 代价惩罚系数 (对应原C++中 kino_astar.cpp 的设置)
        self.lambda_heu = 1.0           
        self.traj_forward_penalty = 1.0 
        self.traj_back_penalty = 2.0    
        self.traj_gear_switch_penalty = 10.0 
        self.traj_steer_penalty = 0.5   
        self.traj_steer_change_penalty = 1.5

        # 碰撞检测安全余量
        self.collision_safety_margin = 0.2  # 米
        
        # One-Shot 分析展开参数
        self.shot_distance = 15.0       # 距终点多少米以内触发 RS 曲线连接
        self.max_search_time = 10.0     # 算法超时截断 (秒)

        self.goal_position_tolerance = 0.5  # 目标位置容差 (米)
        self.goal_yaw_tolerance = math.radians(5.0)  # 目标航向角容差 (弧度)

        self.vis_params = {"vis_enable": True,
                           "vis_frequency": 50}          # 是否启用搜索过程可视化 (默认开启),以及可视化刷新频率 (每多少个节点刷新一次)