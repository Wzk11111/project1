import json

from astar import maze_path, weighted_cost
from minimax_ab import TreeNode, solve_minimax, solve_alphabeta
from challenge import GRID, POINTS, POSITIONS, VALUES, distance_matrix, print_distance_matrix, TreasureGame


def test_problem1():
    start = (0, 0)
    goal = (3, 4)
    grid = [
        ['S', '.', '.', 'X', '.'],
        ['.', 'X', '.', '.', '.'],
        ['.', 'X', '.', 'X', '.'],
        ['.', '.', '.', 'X', 'E'],
        ['X', 'X', '.', '.', '.'],
    ]
    path = maze_path(start, goal, grid)
    expected = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]
    print('题目一 迷宫寻路:')
    print('  路径:', path)
    assert path == expected, f'题目一输出与预期不一致: {path}'
    print('  ✓ 与预期输出一致')

    grid2 = [
        ['S', '.'],
        ['.', 'X'],
    ]
    path2 = maze_path((0, 0), (1, 1), grid2)
    assert path2 == [], f'无解时应返回空列表，实际 {path2}'
    print('  无解返回空列表 ✓')


def test_problem2():
    start = (0, 0)
    goal = (4, 4)
    grid = [
        ['S', '1', '1', '8', '1'],
        ['1', '8', '3', '1', '1'],
        ['1', '8', '1', '8', '1'],
        ['1', '1', '1', '8', '1'],
        ['8', '8', '1', '1', 'E'],
    ]
    cost = weighted_cost(start, goal, grid)
    print('题目二 加权地形:')
    print('  最小总代价:', cost)
    assert cost == 7, f'题目二输出与预期不一致: {cost}'
    print('  ✓ 与预期输出 7 一致')

    grid2 = [
        ['S', 'X'],
        ['X', 'E'],
    ]
    cost2 = weighted_cost((0, 0), (1, 1), grid2)
    assert cost2 == -1, f'无解时应返回 -1，实际 {cost2}'
    print('  无解返回 -1 ✓')


def build_problem3_tree():
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
    return root


def test_problem3():
    root = build_problem3_tree()
    v_minimax, n_minimax = solve_minimax(root)
    v_ab, n_ab = solve_alphabeta(root)
    print('题目三 极大极小博弈树:')
    print(f'  Minimax   最佳得分 = {v_minimax}, 访问节点 = {n_minimax}')
    print(f'  Alpha-Beta 最佳得分 = {v_ab},   访问节点 = {n_ab}')
    assert v_minimax == 5 and v_ab == 5, f'预期最佳得分为 5，实际 {v_minimax}/{v_ab}'
    assert n_ab <= n_minimax, 'Alpha-Beta 访问节点数应不超过 Minimax'
    print('  ✓ 与预期输出 5 一致，且剪枝减少了访问节点数')


def test_problem4():
    mat = distance_matrix()
    print('题目四 竞争环境下的宝藏分配:')
    print('  6×6 距离矩阵（点位顺序:', ' '.join(POINTS), '）:')
    print('  ' + print_distance_matrix(mat).replace('\n', '\n  '))

    game = TreasureGame(mat)
    for H in (2, 4):
        target, value, visited = game.solve(H, use_alphabeta=True)
        result = json.dumps({'best_target': target, 'best_value': value}, ensure_ascii=False)
        print(f'  H = {H}  =>  {result}    访问节点={visited}')

    t2, v2, _ = game.solve(2, use_alphabeta=True)
    t4, v4, _ = game.solve(4, use_alphabeta=True)
    assert t2 == 'T2' and v2 == -1, f'H=2 预期 T2/-1，实际 {t2}/{v2}'
    assert t4 == 'T3' and v4 == 3, f'H=4 预期 T3/3，实际 {t4}/{v4}'
    print('  ✓ H=2、H=4 均与预期输出一致')


if __name__ == '__main__':
    test_problem1()
    test_problem2()
    test_problem3()
    test_problem4()
    print()
    print('=' * 40)
    print('全部测试用例通过 SUCCESS')
