import json
import random

from astar import maze_path, weighted_cost
from minimax_ab import TreeNode, solve_minimax, solve_alphabeta
from challenge import GRID, POINTS, POSITIONS, VALUES, distance_matrix, TreasureGame

random.seed(42)

RESULTS = {}


def _fmt_time(t):
    return f'{t * 1000:.3f} ms'


def gen_maze(n, obstacle_p, seed):
    rng = random.Random(seed)
    grid = []
    for r in range(n):
        row = []
        for c in range(n):
            row.append('X' if rng.random() < obstacle_p else '.')
        grid.append(row)
    grid[0][0] = 'S'
    grid[n - 1][n - 1] = 'E'
    return grid, (0, 0), (n - 1, n - 1)


def gen_weighted(n, seed, lo=1, hi=9):
    rng = random.Random(seed)
    grid = []
    for r in range(n):
        row = []
        for c in range(n):
            row.append(str(rng.randint(lo, hi)))
        grid.append(row)
    grid[0][0] = 'S'
    grid[n - 1][n - 1] = 'E'
    return grid, (0, 0), (n - 1, n - 1)


def run_problem1():
    cases = []

    given_grid = [
        ['S', '.', '.', 'X', '.'],
        ['.', 'X', '.', '.', '.'],
        ['.', 'X', '.', 'X', '.'],
        ['.', '.', '.', 'X', 'E'],
        ['X', 'X', '.', '.', '.'],
    ]
    cases.append(('给定迷宫 5×5', given_grid, (0, 0), (3, 4)))

    for seed in (1, 2):
        g, s, e = gen_maze(30, 0.25, seed)
        cases.append((f'随机迷宫30×30(seed={seed})', g, s, e))

    rows = []
    for name, grid, s, e in cases:
        for h in ('zero', 'manhattan'):
            stats = {}
            maze_path(s, e, grid, h_choice=h, stats=stats)
            rows.append({
                '场景': name, '启发函数': 'h=0(Dijkstra)' if h == 'zero' else '曼哈顿距离',
                '扩展节点数': stats['expanded'], '耗时ms': round(stats['time'] * 1000, 3),
            })
    RESULTS['题目一_启发对照'] = rows


def run_problem2():
    cases = []

    given_grid = [
        ['S', '1', '1', '8', '1'],
        ['1', '8', '3', '1', '1'],
        ['1', '8', '1', '8', '1'],
        ['1', '1', '1', '8', '1'],
        ['8', '8', '1', '1', 'E'],
    ]
    cases.append(('给定加权地图 5×5', given_grid, (0, 0), (4, 4)))

    for seed in (3, 4):
        g, s, e = gen_weighted(30, seed)
        cases.append((f'随机加权地图30×30(seed={seed})', g, s, e))

    for seed in (5,):
        g, s, e = gen_weighted(30, seed, lo=1, hi=3)
        cases.append((f'低代价地图30×30(seed={seed},代价1-3)', g, s, e))

    rows = []
    for name, grid, s, e in cases:
        for h in ('zero', 'manhattan', 'euclidean'):
            stats = {}
            weighted_cost(s, e, grid, h_choice=h, stats=stats)
            label = {'zero': 'h=0(Dijkstra)', 'manhattan': '曼哈顿距离', 'euclidean': '欧氏距离'}[h]
            rows.append({
                '场景': name, '启发函数': label,
                '扩展节点数': stats['expanded'], '耗时ms': round(stats['time'] * 1000, 3),
            })
    RESULTS['题目二_启发对照'] = rows


def _make_random_tree(depth, branch, rng):
    if depth == 0:
        return TreeNode(value=rng.randint(0, 20))
    node = TreeNode(children=[])
    for _ in range(branch):
        node.children.append(_make_random_tree(depth - 1, branch, rng))
    return node


def subtree_value(node, is_max):
    if not node.children:
        return node.value
    vals = [subtree_value(c, not is_max) for c in node.children]
    return max(vals) if is_max else min(vals)


def best_order_tree(node, is_max=True):
    new_node = TreeNode(value=node.value, children=[])
    scored = []
    for c in node.children:
        sub = best_order_tree(c, not is_max)
        scored.append((subtree_value(sub, not is_max), sub))
    scored.sort(key=lambda x: x[0], reverse=is_max)
    new_node.children = [s[1] for s in scored]
    return new_node


def run_problem3():
    rows = []
    order_rows = []

    leaf1 = TreeNode(value=3)
    leaf2 = TreeNode(value=5)
    leaf3 = TreeNode(value=6)
    leaf4 = TreeNode(value=9)
    leaf5 = TreeNode(value=1)
    leaf6 = TreeNode(value=2)
    max1 = TreeNode(children=[leaf1, leaf2])
    max2 = TreeNode(children=[leaf3, leaf4])
    max3 = TreeNode(children=[leaf5, leaf6])
    min1 = TreeNode(children=[max1, max2])
    min2 = TreeNode(children=[max2, max3])
    root = TreeNode(children=[min1, min2])

    v_mm, n_mm = solve_minimax(root)
    v_ab, n_ab = solve_alphabeta(root)
    rows.append({
        '树': '给定树，6叶12节点', '算法': 'Minimax', '最佳得分': v_mm, '访问节点数': n_mm,
    })
    rows.append({
        '树': '给定树，6叶12节点', '算法': 'Alpha-Beta', '最佳得分': v_ab, '访问节点数': n_ab,
    })

    rng = random.Random(7)
    for depth, branch in ((3, 4), (4, 3)):
        tree = _make_random_tree(depth, branch, rng)
        v_mm, n_mm = solve_minimax(tree)
        v_ab, n_ab = solve_alphabeta(tree)
        rows.append({
            '树': f'随机树(深度{depth}×分支{branch})', '算法': 'Minimax',
            '最佳得分': v_mm, '访问节点数': n_mm,
        })
        rows.append({
            '树': f'随机树(深度{depth}×分支{branch})', '算法': 'Alpha-Beta',
            '最佳得分': v_ab, '访问节点数': n_ab,
        })

        ordered = best_order_tree(tree)
        v_ord, n_ord = solve_alphabeta(ordered)
        order_rows.append({
            '树': f'随机树(深度{depth}×分支{branch})',
            '访问顺序': '从左到右', '访问节点数': n_ab,
        })
        order_rows.append({
            '树': f'随机树(深度{depth}×分支{branch})',
            '访问顺序': '最佳排序', '访问节点数': n_ord,
        })

    RESULTS['题目三_剪枝对照'] = rows
    RESULTS['题目三_访问顺序'] = order_rows


def run_problem4():
    mat = distance_matrix()
    RESULTS['题目四_距离矩阵'] = {
        'points': POINTS,
        'positions': POSITIONS,
        'values': VALUES,
        'matrix': mat,
    }

    rows = []
    game = TreasureGame(mat)
    for H in (2, 4):
        for use_ab, name in ((False, 'Minimax'), (True, 'Alpha-Beta')):
            target, value, visited = game.solve(H, use_alphabeta=use_ab)
            rows.append({
                '深度H': H, '算法': name, '最优首步': target,
                '最优收益差U': value, '访问节点数': visited,
            })
    RESULTS['题目四_深度与剪枝对照'] = rows

    order_rows = []
    for key, label in (
        (None, '字典序(从左到右)'),
        (lambda t: (-VALUES[t], t), '按价值降序'),
    ):
        g = TreasureGame(mat, order_key=key)
        for H in (2, 4):
            target, value, visited = g.solve(H, use_alphabeta=True)
            order_rows.append({
                '深度H': H, '访问顺序': label,
                '最优首步': target, '最优收益差U': value, '访问节点数': visited,
            })
    RESULTS['题目四_访问顺序'] = order_rows


def run_admissibility_check():
    maze_ok = 0
    maze_total = 50
    for seed in range(maze_total):
        g, s, e = gen_maze(30, 0.25, seed)
        p0 = maze_path(s, e, g, h_choice='zero')
        pm = maze_path(s, e, g, h_choice='manhattan')
        if len(p0) == len(pm):
            maze_ok += 1
    w_ok = 0
    w_total = 50
    for seed in range(w_total):
        g, s, e = gen_weighted(30, seed, lo=1, hi=9)
        c0 = weighted_cost(s, e, g, h_choice='zero')
        cm = weighted_cost(s, e, g, h_choice='manhattan')
        if c0 == cm:
            w_ok += 1
    RESULTS['可采纳性验证'] = {
        '迷宫_一致数': maze_ok, '迷宫_总数': maze_total,
        '加权_一致数': w_ok, '加权_总数': w_total,
    }


def main():
    run_problem1()
    run_problem2()
    run_problem3()
    run_problem4()
    run_admissibility_check()

    with open('experiment_data.json', 'w', encoding='utf-8') as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2)

    lines = ['实验二 性能分析实验结果', '=' * 50]
    for section, rows in RESULTS.items():
        lines.append('')
        lines.append(f'【{section}】')
        if section == '题目四_距离矩阵':
            lines.append('距离矩阵:')
            lines.append('  点位   ' + '  '.join(rows['points']))
            for i, row in enumerate(rows['matrix']):
                lines.append(f'  {rows["points"][i]:<6}' + '  '.join(f'{v:>3}' for v in row))
            continue
        if isinstance(rows, list):
            keys = list(rows[0].keys())
            lines.append('  ' + ' | '.join(keys))
            for r in rows:
                lines.append('  ' + ' | '.join(str(r[k]) for k in keys))
        elif isinstance(rows, dict):
            for k, v in rows.items():
                lines.append(f'  {k}: {v}')
    with open('results.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))
    print()
    print('实验结果已写入 results.txt 与 experiment_data.json')


if __name__ == '__main__':
    main()
