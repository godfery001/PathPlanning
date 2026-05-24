import numpy as np
import heapq

class HeuristicMap:
    def __init__(self, env):
        self.env = env
        self.dist_map = np.full((env.grid_size_x, env.grid_size_y), float('inf'))

    def update_goal(self, goal_state):
        self.dist_map.fill(float('inf'))
        goal_x, goal_y, _ = goal_state
        idx_g = int((goal_x - self.env.origin_x) / self.env.cfg.xy_resolution)
        idy_g = int((goal_y - self.env.origin_y) / self.env.cfg.xy_resolution)
        
        if not (0 <= idx_g < self.env.grid_size_x and 0 <= idy_g < self.env.grid_size_y): return
            
        pq = [(0.0, idx_g, idy_g)]
        self.dist_map[idx_g, idy_g] = 0.0
        directions = [(1,0,1.0), (-1,0,1.0), (0,1,1.0), (0,-1,1.0), 
                      (1,1,1.414), (-1,-1,1.414), (-1,1,1.414), (1,-1,1.414)]
                      
        while pq:
            cost, x, y = heapq.heappop(pq)
            if cost > self.dist_map[x, y]: continue
            for dx, dy, move_cost in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.env.grid_size_x and 0 <= ny < self.env.grid_size_y:
                    if self.env.obstacle_grid[nx, ny] == 0:
                        new_cost = cost + move_cost * self.env.cfg.xy_resolution
                        if new_cost < self.dist_map[nx, ny]:
                            self.dist_map[nx, ny] = new_cost
                            heapq.heappush(pq, (new_cost, nx, ny))

    def get_heuristic(self, state):
        x, y, _ = state
        idx = int((x - self.env.origin_x) / self.env.cfg.xy_resolution)
        idy = int((y - self.env.origin_y) / self.env.cfg.xy_resolution)
        if 0 <= idx < self.env.grid_size_x and 0 <= idy < self.env.grid_size_y:
            return self.dist_map[idx, idy]
        return float('inf')