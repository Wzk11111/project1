from astar import _WeightedGrid, _astar, manhattan

GRID = [
    ['A', '1', '1', 'X', '2', '1', 'T1'],
    ['1', 'X', '6', 'X', '1', 'X', '1'],
    ['1', '1', 'T2', '1', '1', '1', '1'],
    ['X', '1', 'X', '8', 'X', '1', 'X'],
    ['1', '1', '1', '1', 'T3', '1', '1'],
    ['1', 'X', '9', 'X', '1', 'X', '1'],
    ['T4', '1', '1', 'X', '1', '1', 'B'],
]

POINTS = ['A', 'B', 'T1', 'T2', 'T3', 'T4']

POSITIONS = {
    'A': (0, 0),
    'B': (6, 6),
    'T1': (0, 6),
    'T2': (2, 2),
    'T3': (4, 4),
    'T4': (6, 0),
}

VALUES = {'T1': 9, 'T2': 6, 'T3': 7, 'T4': 7}

LETTER_COST = 1


class _TreasureGrid(_WeightedGrid):

    def cost(self, r, c):
        ch = str(self.grid[r][c])
        if ch == 'X':
            raise ValueError('障碍物不可通行')
        if ch.isdigit():
            return int(ch)
        return LETTER_COST


def distance_matrix(verbose=False):
    grid = _TreasureGrid(GRID)
    n = len(POINTS)
    mat = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i][j] = 0
            else:
                cost, path, _ = _astar(
                    POSITIONS[POINTS[i]], POSITIONS[POINTS[j]], grid, manhattan
                )
                mat[i][j] = cost
    return mat


def print_distance_matrix(mat):
    header = '点位   ' + '  '.join(POINTS)
    lines = [header]
    for i, row in enumerate(mat):
        lines.append(f'{POINTS[i]:<6}' + '  '.join(f'{v:>3}' for v in row))
    return '\n'.join(lines)


class TreasureGame:

    def __init__(self, distances, values=None, order_key=None):
        self.dist = distances
        self.values = values if values is not None else VALUES
        self.point_index = {p: i for i, p in enumerate(POINTS)}
        self.order_key = order_key

    def initial_state(self):
        return (
            'A',
            'B',
            frozenset(['T1', 'T2', 'T3', 'T4']),
            0.0,
            0.0,
        )

    def apply(self, state, treasure):
        a_pos, b_pos, remaining, sa, sb = state
        turn_a = self._is_a_turn(state)
        pos = a_pos if turn_a else b_pos
        gain = self.values[treasure] - self.dist[self.point_index[pos]][self.point_index[treasure]]
        new_remaining = remaining - {treasure}
        if turn_a:
            return (treasure, b_pos, new_remaining, sa + gain, sb)
        return (a_pos, treasure, new_remaining, sa, sb + gain)

    @staticmethod
    def _is_a_turn(state):
        return len(state[2]) % 2 == 0

    def terminal(self, state, depth, H):
        return depth == H or len(state[2]) == 0

    def utility(self, state):
        return state[3] - state[4]

    def minimax(self, state, depth, H, stats=None):
        if stats is None:
            stats = {}
        stats['visited'] = stats.get('visited', 0) + 1
        if self.terminal(state, depth, H):
            return self.utility(state), None
        turn_a = self._is_a_turn(state)
        best = float('-inf') if turn_a else float('inf')
        best_action = None
        order = sorted(state[2]) if self.order_key is None else sorted(state[2], key=self.order_key)
        for t in order:
            child = self.apply(state, t)
            v, _ = self.minimax(child, depth + 1, H, stats)
            if turn_a:
                if v > best:
                    best, best_action = v, t
            else:
                if v < best:
                    best, best_action = v, t
        return best, best_action

    def alphabeta(self, state, depth, H, alpha, beta, stats=None):
        if stats is None:
            stats = {}
        stats['visited'] = stats.get('visited', 0) + 1
        if self.terminal(state, depth, H):
            return self.utility(state), None
        turn_a = self._is_a_turn(state)
        best = float('-inf') if turn_a else float('inf')
        best_action = None
        order = sorted(state[2]) if self.order_key is None else sorted(state[2], key=self.order_key)
        for t in order:
            child = self.apply(state, t)
            v, _ = self.alphabeta(child, depth + 1, H, alpha, beta, stats)
            if turn_a:
                if v > best:
                    best, best_action = v, t
                alpha = max(alpha, best)
            else:
                if v < best:
                    best, best_action = v, t
                beta = min(beta, best)
            if beta <= alpha:
                break
        return best, best_action

    def solve(self, H, use_alphabeta=True):
        state = self.initial_state()
        stats = {}
        if use_alphabeta:
            value, action = self.alphabeta(
                state, 0, H, float('-inf'), float('inf'), stats
            )
        else:
            value, action = self.minimax(state, 0, H, stats)
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return action, value, stats['visited']


if __name__ == '__main__':
    mat = distance_matrix()
    print(print_distance_matrix(mat))
    game = TreasureGame(mat)
    for H in (2, 4):
        for ab in (False, True):
            name = 'AlphaBeta' if ab else 'Minimax'
            target, value, visited = game.solve(H, use_alphabeta=ab)
            print(f'H={H}  {name:<9} best_target={target}  best_value={value}  访问节点={visited}')
