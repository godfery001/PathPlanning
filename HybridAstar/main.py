import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import time

# 设置内置中文字体，支持 Windows / macOS / Linux 常用中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from PathPlanning.HybridAstar.vehicle import VehicleParam
from PathPlanning.HybridAstar.config import SearchConfig
from PathPlanning.HybridAstar.env import MapInterface
from PathPlanning.HybridAstar.heuristic import HeuristicMap
from PathPlanning.HybridAstar.kino_astar import KinoAstar

def plot_vehicle(ax, state, vp, color='orange', alpha=0.4):
    x, y, yaw = state
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    
    rear_right_x = x - vp.rear_edge_to_center * cos_y + vp.right_edge_to_center * sin_y
    rear_right_y = y - vp.rear_edge_to_center * sin_y - vp.right_edge_to_center * cos_y
    
    rect = patches.Rectangle((rear_right_x, rear_right_y),
                             vp.length, vp.width,
                             angle=math.degrees(yaw),
                             linewidth=1.2, edgecolor=color, facecolor='none', alpha=alpha)
    ax.add_patch(rect)

def main():
    vp = VehicleParam()
    cfg = SearchConfig()
    env = MapInterface(cfg, vp)
    
    # 注入环境障碍物
    # env.add_obstacle_circle(18.0, 20.0, 3.0)
    env.add_obstacle_polygon([[10.0,25.0],[10.0,30.0],[20.0,30.0]])
    env.add_obstacle_polygon([[10.0, 0.0], [12.0, 0.0], [12.0, 10.0], [10.0, 10.0]])
    env.add_obstacle_polygon([[30.0, 15.0], [35.0, 15.0], [35.0, 30.0], [32.0, 28.0], [30.0, 15.0]])
    
    h_map = HeuristicMap(env)
    planner = KinoAstar(cfg, vp, env, h_map)
    
    start_state = (5.0, 5.0, math.radians(45.0))
    goal_state = (40.0, 28.0, math.radians(60.0))
    
    # 【1. 开启交互模式，准备动态绘图】
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-5, cfg.map_size_x)
    ax.set_ylim(-5, cfg.map_size_y)
    
    # 提前绘制黑色障碍物与起终点
    for i in range(env.grid_size_x):
        for j in range(env.grid_size_y):
            if env.obstacle_grid[i, j] == 1:
                gx = env.origin_x + i * cfg.xy_resolution
                gy = env.origin_y + j * cfg.xy_resolution
                ax.add_patch(patches.Rectangle((gx, gy), cfg.xy_resolution, cfg.xy_resolution, color='black'))
                
    plot_vehicle(ax, start_state, vp, color='green', alpha=1.0)
    plot_vehicle(ax, goal_state, vp, color='blue', alpha=1.0)
    ax.plot(start_state[0], start_state[1], 'gx', markersize=8)
    ax.plot(goal_state[0], goal_state[1], 'bx', markersize=8)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.title("Rigorous Hybrid A* w/ Dynamic Expansion")
    plt.draw() # 初始画面渲染

    # 【2. 将 ax 传入，开始动态展示搜索过程】
    print("开始路径搜索...")
    path = planner.search(start_state, goal_state, ax=ax)
    
    # 【3. 搜索结束，绘制最终动画与轨迹】
    if path:

        finale_state = path[-1]
        pos_error = math.hypot(finale_state[0] - goal_state[0], finale_state[1] - goal_state[1])
        yaw_error = abs((finale_state[2] - goal_state[2] + math.pi) % (2 * math.pi) - math.pi)
        print(f"路径搜索完成！最终位置误差: {pos_error:.2f} m, 航向误差: {math.degrees(yaw_error):.2f} deg")
        
        # 将中文字符显示在终点状态的上方一点，不与标题重合，添加半透明白色背景框增加可读性
        ax.text(cfg.map_size_x, cfg.map_size_y, 
                f"位置误差: {pos_error:.2f} m\n航向误差: {math.degrees(yaw_error):.2f} deg",
                ha='center', va='bottom', fontsize=11, color='red',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', boxstyle='round,pad=0.5'))
        
        # 绘制最终红线
        ax.plot([s[0] for s in path], [s[1] for s in path], 'r-', linewidth=2.0)
        
        # 模拟车辆行驶动画
        step = max(1, len(path) // 25)
        for i in range(0, len(path), step):
            plot_vehicle(ax, path[i], vp, color='orange', alpha=0.6)
            plt.pause(0.1)
    else:
        print("未找到有效路径！")
            
    plt.ioff() # 关闭交互模式
    plt.show() # 保持窗口常亮

if __name__ == "__main__":
    main()