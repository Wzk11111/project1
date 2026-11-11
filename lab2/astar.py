import heapq
import time

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def zero_heuristic(a, b):
    return 0.0


HEURISTICS = {
    'manhattan': manhattan,
    'euclidean': euclidean,
    'zero': zero_heuristic,
}


def _astar(start, goal, step_cost, heuristic, stats=None):
    R = step_cost.rows
    C = step_cost.cols

    def in_bounds(r, c):
        return 0 <= r < R and 0 <= c < C

    open_heap = [(heuristic(start, goal), 0, start)]
    g_score = {start: 0}
    came_from = {}
    closed = set()
    expanded = 0

    while open_heap:
        f, g, node = heapq.heappop(open_heap)
        if node in closed:
            continue
        closed.add(node)
        expanded += 1
        if node == goal:
            path = []
            cur = node
            while cur is not None:
                path.append(cur)
                cur = came_from.get(cur)
            path.reverse()
            return g, path, expanded
        r, c = node
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                continue
            if not step_cost.passable(nr, nc):
                continue
            neighbor = (nr, nc)
            step = step_cost.cost(nr, nc)
            tentative_g = g + step
            if neighbor in closed and g_score.get(neighbor, float('inf')) <= tentative_g:
                continue
            if tentative_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = node
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f, tentative_g, neighbor))
    return None, [], expanded


class _MazeGrid:

    def __init__(self, grid):
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        self.grid = grid

    def passable(self, r, c):
        ch = self.grid[r][c]
        return ch != 'X'

    def cost(self, r, c):
        return 1


class _WeightedGrid:

    def __init__(self, grid):
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        self.grid = grid

    def passable(self, r, c):
        ch = str(self.grid[r][c])
        return ch != 'X'

    def cost(self, r, c):
        ch = str(self.grid[r][c])
        if ch in ('S', 'E'):
            return 0
        return int(ch)


def maze_path(start, goal, grid, h_choice='manhattan', stats=None):
    step_cost = _MazeGrid(grid)
    h = HEURISTICS[h_choice]
    t0 = time.perf_counter()
    total_cost, path, expanded = _astar(start, goal, step_cost, h, stats)
    elapsed = time.perf_counter() - t0
    if stats is not None:
        stats['expanded'] = expanded
        stats['time'] = elapsed
    return path


def weighted_cost(start, goal, grid, h_choice='manhattan', stats=None):
    step_cost = _WeightedGrid(grid)
    h = HEURISTICS[h_choice]
    t0 = time.perf_counter()
    total_cost, path, expanded = _astar(start, goal, step_cost, h, stats)
    elapsed = time.perf_counter() - t0
    if stats is not None:
        stats['expanded'] = expanded
        stats['time'] = elapsed
    return -1 if total_cost is None else total_cost
