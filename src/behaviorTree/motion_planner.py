import numpy as np

class Node:
    def __init__(self, pos):
        self.pos = np.array(pos)
        self.parent = None

class TaskSpaceRRT:
    def __init__(self, step_size=0.05, max_iter=2000, bounds=([-0.5, -0.5, 0.75], [0.5, 0.5, 1.3])):
        self.step_size = step_size
        self.max_iter = max_iter
        self.bounds = np.array(bounds)
        
    def _is_collision_free(self, p1, p2, obstacles, resolution=0.01):
        dist = np.linalg.norm(p2 - p1)
        steps = max(2, int(dist / resolution))
        for i in range(steps + 1):
            t = i / steps
            p = p1 + t * (p2 - p1)
            for obs in obstacles:
                b_min, b_max = obs['min'], obs['max']
                if np.all(p >= b_min) and np.all(p <= b_max):
                    return False
        return True

    def _get_random_point(self, goal, goal_bias=0.2):
        # Goal oriented RRT
        # With probability 'goal_bias' it samples the goal point, else it samples a random bounded point
        if np.random.rand() < goal_bias:
            return np.array(goal)
        return np.random.uniform(self.bounds[0], self.bounds[1])

    def plan(self, start, goal, obstacles):
        # Path planning in task space using RRT
        start_node = Node(start)
        nodes = [start_node]
        
        for _ in range(self.max_iter):
            rand_pos = self._get_random_point(goal)
            
            # Find nearest node
            nearest_node = min(nodes, key=lambda n: np.linalg.norm(n.pos - rand_pos))
            
            # Steer
            dir_vec = rand_pos - nearest_node.pos
            dist = np.linalg.norm(dir_vec)
            if dist > self.step_size:
                dir_vec = dir_vec / dist * self.step_size
            new_pos = nearest_node.pos + dir_vec
            
            # Check collision
            if self._is_collision_free(nearest_node.pos, new_pos, obstacles):
                new_node = Node(new_pos)
                new_node.parent = nearest_node
                nodes.append(new_node)
                
                # Check if reached goal
                if np.linalg.norm(new_pos - goal) <= self.step_size:
                    if self._is_collision_free(new_pos, goal, obstacles):
                        goal_node = Node(goal)
                        goal_node.parent = new_node
                        nodes.append(goal_node)
                        
                        # Reconstruct path
                        path = []
                        curr = goal_node
                        while curr is not None:
                            path.append(curr.pos)
                            curr = curr.parent
                        raw_path = path[::-1]
                        return self._shortcut_path(raw_path, obstacles)
                        
        # Fallback to direct path if RRT fails
        return [np.array(start), np.array(goal)]

    def _shortcut_path(self, path, obstacles):
        # Search backwards from the goal to remove unnecessary waypoints
        # while maintaining collision-free path
        if len(path) <= 2:
            return path
        shortcutted = [path[0]]
        curr_idx = 0
        while curr_idx < len(path) - 1:
            next_idx = len(path) - 1
            while next_idx > curr_idx + 1:
                if self._is_collision_free(path[curr_idx], path[next_idx], obstacles):
                    break
                next_idx -= 1
            shortcutted.append(path[next_idx])
            curr_idx = next_idx
        return shortcutted

