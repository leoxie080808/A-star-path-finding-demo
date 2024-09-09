#-----------------
# Created by Leo Xie 
# A* Demo visual tool, initially meant for ROS
# robot, later ported to pygame for visuals 
#-----------------

#Custom queue removed for simpler built-in priority queue

import pygame
import numpy as np
import math
from queue import PriorityQueue
from pygame import gfxdraw

# Define the colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GREY = (128, 128, 128)
PURPLE = (128, 0, 128)

# Set the dimensions
WIDTH = 800
GRID_SIZE = 50
TILE_SIZE = WIDTH // GRID_SIZE

# Initialize Pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((WIDTH, WIDTH + 50))  # Extra 50px for instructions
pygame.display.set_caption("A* Pathfinding")

# Font for instructions
font = pygame.font.SysFont("Arial", 18)

# Heuristic function (Diagonal distance)
def heuristic(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy)

# A* algorithm using Priority Queue
def astar(draw, grid, start, goal):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {cell: float("inf") for row in grid for cell in row}
    g_score[start] = 0
    f_score = {cell: float("inf") for row in grid for cell in row}
    f_score[start] = heuristic(start.get_pos(), goal.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == goal:
            reconstruct_path(came_from, goal, draw)
            goal.make_goal()
            start.make_start()
            return True, came_from

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + heuristic(neighbor.get_pos(), goal.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False, came_from

# Function to reconstruct the path
def reconstruct_path(came_from, current, draw):
    path = []
    while current in came_from:
        current = came_from[current]
        current.make_path()
        path.append(current.get_pos())
        draw()
    return path

# Draw a smooth curve through the points
def draw_curved_path(screen, path):
    if len(path) < 3:
        return
    # Create a smoothed version of the path
    for i in range(1, len(path) - 1):
        # Get current point and adjacent points
        x1, y1 = path[i-1]
        x2, y2 = path[i]
        x3, y3 = path[i+1]

        # Find a midpoint and draw an arc from the middle of the grid cell
        midpoint = ((x1 + x3) * TILE_SIZE // 2, (y1 + y3) * TILE_SIZE // 2)

        # Use pygame.gfxdraw to draw smoother curves
        pygame.draw.line(screen, RED, (x2 * TILE_SIZE + TILE_SIZE // 2, y2 * TILE_SIZE + TILE_SIZE // 2), midpoint, 3)
        pygame.display.update()

# Define the Node class to represent each cell
class Node:
    def __init__(self, row, col, width):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == YELLOW

    def is_open(self):
        return self.color == BLUE

    def is_obstacle(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == GREEN

    def is_goal(self):
        return self.color == RED

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = GREEN

    def make_closed(self):
        self.color = YELLOW

    def make_open(self):
        self.color = BLUE

    def make_obstacle(self):
        self.color = BLACK

    def make_goal(self):
        self.color = RED

    def make_path(self):
        self.color = PURPLE

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        # Vertical and Horizontal Neighbors
        if self.row < GRID_SIZE - 1 and not grid[self.row + 1][self.col].is_obstacle():  # DOWN
            self.neighbors.append(grid[self.row + 1][self.col])
        if self.row > 0 and not grid[self.row - 1][self.col].is_obstacle():  # UP
            self.neighbors.append(grid[self.row - 1][self.col])
        if self.col < GRID_SIZE - 1 and not grid[self.row][self.col + 1].is_obstacle():  # RIGHT
            self.neighbors.append(grid[self.row][self.col + 1])
        if self.col > 0 and not grid[self.row][self.col - 1].is_obstacle():  # LEFT
            self.neighbors.append(grid[self.row][self.col - 1])

        # Diagonal Neighbors
        if self.row > 0 and self.col > 0 and not grid[self.row - 1][self.col - 1].is_obstacle():  # UP-LEFT
            self.neighbors.append(grid[self.row - 1][self.col - 1])
        if self.row > 0 and self.col < GRID_SIZE - 1 and not grid[self.row - 1][self.col + 1].is_obstacle():  # UP-RIGHT
            self.neighbors.append(grid[self.row - 1][self.col + 1])
        if self.row < GRID_SIZE - 1 and self.col > 0 and not grid[self.row + 1][self.col - 1].is_obstacle():  # DOWN-LEFT
            self.neighbors.append(grid[self.row + 1][self.col - 1])
        if self.row < GRID_SIZE - 1 and self.col < GRID_SIZE - 1 and not grid[self.row + 1][self.col + 1].is_obstacle():  # DOWN-RIGHT
            self.neighbors.append(grid[self.row + 1][self.col + 1])

# Create the grid
def make_grid():
    grid = []
    for i in range(GRID_SIZE):
        grid.append([])
        for j in range(GRID_SIZE):
            node = Node(i, j, TILE_SIZE)
            grid[i].append(node)
    return grid

# Draw the grid lines
def draw_grid(screen):
    for i in range(GRID_SIZE):
        pygame.draw.line(screen, BLACK, (0, i * TILE_SIZE), (WIDTH, i * TILE_SIZE))
        pygame.draw.line(screen, BLACK, (i * TILE_SIZE, 0), (i * TILE_SIZE, WIDTH))

# Draw the entire grid with the nodes
def draw(screen, grid):
    screen.fill(WHITE)
    for row in grid:
        for node in row:
            node.draw(screen)
    draw_grid(screen)
    pygame.display.update()

# Get the position of the mouse click in terms of the grid
def get_clicked_pos(pos):
    y, x = pos
    row = y // TILE_SIZE
    col = x // TILE_SIZE
    return row, col

# Main game loop
def main():
    grid = make_grid()

    start = None
    goal = None

    running = True
    while running:
        draw(screen, grid)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if pygame.mouse.get_pressed()[0]:  # Left click
                pos = pygame.mouse.get_pos()
                if pos[1] >= WIDTH:  # Ignore clicks on the instructions area
                    continue
                row, col = get_clicked_pos(pos)
                if row < 0 or row >= GRID_SIZE or col < 0 or col >= GRID_SIZE:
                    continue  # Ignore out-of-bounds clicks

                node = grid[row][col]
                if not start and node != goal:
                    start = node
                    start.make_start()
                elif not goal and node != start:
                    goal = node
                    goal.make_goal()
                elif node != goal and node != start:
                    node.make_obstacle()

            elif pygame.mouse.get_pressed()[2]:  # Right click to reset
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos)
                if row < 0 or row >= GRID_SIZE or col < 0 or col >= GRID_SIZE:
                    continue  # Ignore out-of-bounds clicks
                node = grid[row][col]
                node.reset()
                if node == start:
                    start = None
                elif node == goal:
                    goal = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and start and goal:
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid)

                    success, came_from = astar(lambda: draw(screen, grid), grid, start, goal)
                    if success:
                        path = reconstruct_path(came_from, goal, lambda: draw(screen, grid))  # Pass the draw function
                        draw_curved_path(screen, path)


    pygame.quit()

if __name__ == "__main__":
    main()

