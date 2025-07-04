import math
from queue import PriorityQueue

# Heuristic function (Diagonal distance)
def heuristic(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy)

# Node class representing a cell in the grid
class Node:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.neighbors = []
        self.is_obstacle = False

    def get_pos(self):
        return self.row, self.col

    def update_neighbors(self, grid, grid_size):
        self.neighbors = []
        r, c = self.row, self.col

        directions = [  # 8 directions: up, down, left, right, and diagonals
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                neighbor = grid[nr][nc]
                if not neighbor.is_obstacle:
                    self.neighbors.append(neighbor)

# A* search algorithm (no GUI)
def astar(grid, start, goal):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {node: float("inf") for row in grid for node in row}
    f_score = {node: float("inf") for row in grid for node in row}

    g_score[start] = 0
    f_score[start] = heuristic(start.get_pos(), goal.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == goal:
            return reconstruct_path(came_from, goal)

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1  # cost = 1 for all valid moves
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + heuristic(neighbor.get_pos(), goal.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)

    return None  # No path found

# Reconstruct the path from goal to start
def reconstruct_path(came_from, current):
    path = [current.get_pos()]
    while current in came_from:
        current = came_from[current]
        path.append(current.get_pos())
    path.reverse()
    return path