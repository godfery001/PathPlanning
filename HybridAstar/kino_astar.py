import math
import heapq
import time
import matplotlib.pyplot as plt

try:
    import reeds_shepp
    HAS_RS = True
except ImportError:
    print("警告: 未检测到 reeds_shepp 库，将退化为纯离散搜索。")
    HAS_RS = False

class Node:
    def __init__(self, state, idx, yaw_idx):
        self.state = state        
        self.idx = idx            
        self.yaw_idx = yaw_idx
        self.g_score = float('inf')
        self.f_score = float('inf')
        self.parent = None
        self.input = (0.0, 0.0)   
        self.singul = 1           

    def __lt__(self, other):
        return self.f_score < other.f_score

class KinoAstar:
    def __init__(self, cfg, vp, env, heuristic_map):
        self.cfg = cfg
        self.vp = vp
        self.env = env
        self.h_map = heuristic_map
        self.yaw_origin = -math.pi
        self.vis_enable=self.cfg.vis_params.get("vis_enable", True)
        self.vis_frequency=self.cfg.vis_params.get("vis_frequency", 50)

    def normalize_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def pos_to_index(self, pos):
        idx = int((pos[0] - self.env.origin_x) / self.cfg.xy_resolution)
        idy = int((pos[1] - self.env.origin_y) / self.cfg.xy_resolution)
        return (idx, idy)

    def yaw_to_index(self, yaw):
        yaw = self.normalize_angle(yaw)
        return math.floor((yaw - self.yaw_origin) / self.cfg.yaw_resolution)

    def state_transit(self, state0, ctrl_input):
        steer, arc = ctrl_input
        x0, y0, yaw0 = state0
        if abs(steer) > 1e-4:
            k = math.tan(steer) / self.vp.wheel_base
            x1 = x0 + (math.sin(yaw0 + arc * k) - math.sin(yaw0)) / k
            y1 = y0 - (math.cos(yaw0 + arc * k) - math.cos(yaw0)) / k
            yaw1 = self.normalize_angle(yaw0 + arc * k)
        else:
            x1 = x0 + arc * math.cos(yaw0)
            y1 = y0 + arc * math.sin(yaw0)
            yaw1 = yaw0
        return (x1, y1, yaw1)

    def analytic_expansion(self, cur_state, goal_state):
        if not HAS_RS: return [], 0.0, False
        try:
            step_size = self.cfg.xy_resolution
            print(f"尝试 RS 曲线连接... (距离: {math.hypot(cur_state[0] - goal_state[0], cur_state[1] - goal_state[1]):.2f} m)")
            rs_path = reeds_shepp.path_sample(cur_state, goal_state, self.vp.min_turning_radius, step_size)
            print(f"RS 曲线连接成功，路径长度: {len(rs_path)} 点")
        except:
            return [], 0.0, False
        
        if rs_path is None:
            return [], 0.0, False
        
        path_list=[(pt[0], pt[1], pt[2]) for pt in rs_path]
        for pt in rs_path:
                state=(pt[0], pt[1], pt[2])
                if self.env.check_collision(state):
                    print("RS 曲线碰撞检测失败！")
                    return [], 0.0, False
                # else:                    
                #     print(f"RS 曲线点: ({pt[0]:.2f}, {pt[1]:.2f}, {math.degrees(pt[2]):.1f}°,{pt[3]:.2f},{pt[4]}) - 无碰撞")
                #     length+=pt[4]

        cost, singul = 0.0, 1 
        for i in range(1, len(rs_path)):
            dx, dy = rs_path[i][0] - rs_path[i-1][0], rs_path[i][1] - rs_path[i-1][1]
            dist = math.hypot(dx, dy)
            move_yaw = math.atan2(dy, dx)
            diff = abs(self.normalize_angle(move_yaw - rs_path[i-1][2]))
            cur_singul = 1 if diff < math.pi/2 else -1
            
            cost += dist * (self.cfg.traj_forward_penalty if cur_singul > 0 else self.cfg.traj_back_penalty)
            if cur_singul * singul < 0: cost += self.cfg.traj_gear_switch_penalty
            singul = cur_singul
            
        print(f"RS 曲线总代价: {cost:.2f}")
        return path_list, cost, True

    def retrieve_path(self, end_node):
        path, curr = [], end_node
        while curr is not None:
            path.append(curr.state)
            curr = curr.parent
        return path[::-1]

    # 【新增入参 ax】：用于传递 matplotlib 的坐标轴进行动态绘制
    def search(self, start_state, goal_state, ax=None):
        self.h_map.update_goal(goal_state)

        if self.env.check_collision(start_state):
            print("ERROR: 起点位于障碍物内！")
            return None
        if self.env.check_collision(goal_state):
            print("ERROR: 终点位于障碍物内！")
            return None

        open_set = []
        g_score_table = {}  
        closed_set = set() # 【修复】：严格的闭集，防止节点重复引发指数爆炸
        
        start_idx, start_yaw_idx = self.pos_to_index(start_state), self.yaw_to_index(start_state[2])
        root = Node(start_state, start_idx, start_yaw_idx)
        root.g_score = 0.0
        root.f_score = self.cfg.lambda_heu * self.h_map.get_heuristic(start_state)
        
        heapq.heappush(open_set, root)
        g_score_table[(start_idx, start_yaw_idx)] = root.g_score
        start_time = time.time()
        
        expand_count = 0
        plot_x, plot_y = [], [] # 用于批量绘制扩展节点
        tree_branches_x, tree_branches_y = [], [] # 用于绘制搜索树分支

        while open_set:
            if time.time() - start_time > self.cfg.max_search_time:
                print("FAILED: 搜索超时！")
                return None
                
            cur_node = heapq.heappop(open_set)
            state_key = (cur_node.idx, cur_node.yaw_idx)
            
            # 【修复】：严格验证是否在闭集中
            if state_key in closed_set: continue
            closed_set.add(state_key)
            
            expand_count += 1

            # 【动态可视化扩展过程】
            # if ax is not None:
            #     plot_x.append(cur_node.state[0])
            #     plot_y.append(cur_node.state[1])
            #     if len(plot_x) >= 100:  # 每100个节点刷新一次屏幕，防止卡顿
            #         ax.plot(plot_x, plot_y, '.', color='c', markersize=2, alpha=0.5)
            #         plt.pause(0.001)
            #         plot_x, plot_y = [], []

            if self.vis_enable and ax is not None:
                if cur_node.parent is not None:
                    tree_branches_x.extend([cur_node.parent.state[0], cur_node.state[0], None])
                    tree_branches_y.extend([cur_node.parent.state[1], cur_node.state[1], None])
                    if expand_count % self.vis_frequency == 0: # 每self.vis_frequency(默认50)个节点绘制一次搜索树分支
                        ax.plot(tree_branches_x, tree_branches_y, '-', color='cyan', linewidth=0.5, alpha=0.7)
                        plt.pause(0.001)
                        tree_branches_x, tree_branches_y = [], []
            elif self.vis_enable and ax is None:
                print(f"INFO: 已扩展节点数: {expand_count}，但未提供 ax 进行可视化！")

            # 1. 尝试 Reeds-Shepp 曲线 One-Shot 连接
            if math.hypot(cur_node.state[0] - goal_state[0], cur_node.state[1] - goal_state[1]) < self.cfg.shot_distance and expand_count % 5==0:
                rs_path, _, is_success = self.analytic_expansion(cur_node.state, goal_state)
                if is_success:
                    print(f"SUCCESS: 通过 RS 曲线命中目标! 耗时: {time.time()-start_time:.3f}s")
                    return self.retrieve_path(cur_node) + rs_path[1:]
                # else:
                #     print(f"INFO: RS 曲线连接失败，继续离散搜索... (已扩展节点数: {expand_count})")
                    
            # 2. 【新增】：纯离散逼近容差终点 (如果RS库缺失或失败的保底方案)
            if math.hypot(cur_node.state[0] - goal_state[0], cur_node.state[1] - goal_state[1]) < self.cfg.goal_position_tolerance:
                if abs(self.normalize_angle(cur_node.state[2] - goal_state[2])) < self.cfg.goal_yaw_tolerance:
                    print(f"SUCCESS: 纯离散搜索抵达容差目标点! 耗时: {time.time()-start_time:.3f}s")
                    return self.retrieve_path(cur_node)

            # 3. 拓展子节点
            for arc in [-self.cfg.step_arc, self.cfg.step_arc]:
                steer = -self.vp.max_steering_angle
                while steer <= self.vp.max_steering_angle + 1e-3:
                    singul = 1 if arc > 0 else -1
                    is_occ, intermediate_states = False, []
                    
                    for k in range(1, self.cfg.check_num + 1):
                        xt = self.state_transit(cur_node.state, (steer, arc * (k / self.cfg.check_num)))
                        intermediate_states.append(xt)
                        if self.env.check_collision(xt):
                            is_occ = True
                            break
                            
                    if not is_occ:
                        pro_state = intermediate_states[-1]
                        
                        tmp_g_score = cur_node.g_score + abs(arc) * (self.cfg.traj_forward_penalty if singul > 0 else self.cfg.traj_back_penalty)
                        if singul * cur_node.singul < 0: tmp_g_score += self.cfg.traj_gear_switch_penalty
                        tmp_g_score += self.cfg.traj_steer_penalty * abs(steer) * abs(arc)
                        tmp_g_score += self.cfg.traj_steer_change_penalty * abs(steer - cur_node.input[0])
                        
                        pro_idx, pro_yaw_idx = self.pos_to_index(pro_state), self.yaw_to_index(pro_state[2])
                        pro_key = (pro_idx, pro_yaw_idx)
                        
                        # 仅当代价值更优时，加入OpenList
                        if tmp_g_score < g_score_table.get(pro_key, float('inf')):
                            g_score_table[pro_key] = tmp_g_score
                            pro_node = Node(pro_state, pro_idx, pro_yaw_idx)
                            pro_node.g_score = tmp_g_score
                            
                            # 启发式: 为了加速搜索，给启发式加上略微贪心的权重 (1.5)
                            pro_node.f_score = tmp_g_score + 1.5 * self.cfg.lambda_heu * self.h_map.get_heuristic(pro_state)
                            pro_node.input = (steer, arc)
                            pro_node.parent = cur_node
                            pro_node.singul = singul
                            heapq.heappush(open_set, pro_node)
                            
                    steer += self.cfg.steer_step
                    
        return None