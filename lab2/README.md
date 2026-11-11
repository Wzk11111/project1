GitHub仓库：https://github.com/Wzk11111/project1

# 实验二 搜索算法：A*与Alpha-Beta剪枝

本目录是仓库中的lab2子目录。《当代人工智能》实验二，实现迷宫寻路、加权地形、极大极小博弈树与竞争环境下的宝藏分配四道题目，
并附带启发函数对照、剪枝效率与访问顺序等性能分析实验。

## 环境与依赖

- 仅使用 Python 标准库（`heapq`、`json`、`random`），无第三方依赖。
- 建议 Python 3.8 及以上版本，实际实验环境为 Windows 11 与 Python 3.13，Linux / macOS 同样可运行。

## 目录结构

```
lab2/
├── astar.py          # 通用 A* 实现，同时服务题目一（迷宫）与题目二（加权地形）
├── minimax_ab.py     # 题目三：Minimax 与 Alpha-Beta 剪枝（TreeNode 结构）
├── challenge.py      # 题目四：综合题，距离矩阵 + 宝藏争夺博弈搜索
├── main.py           # 统一测试入口，运行四个题目的给定测试用例
├── experiments.py    # 性能分析实验脚本，生成 results.txt 与 experiment_data.json
├── make_figures.py   # 根据实验数据生成报告图表，输出到 figs 目录
├── figs/             # 报告用的六张 SVG 矢量图
├── README.md         # 本文件
├── report.html       # 实验报告源文件，共4页，未超过5页限制
└── 实验报告.pdf      # 由 report.html 导出的4页PDF
```

## 运行顺序

```bash
# 1. 运行四个题目的给定测试用例并校验预期输出
python main.py

# 2. 运行性能分析实验（启发函数对照、剪枝对照、访问顺序、可采纳性验证）
python experiments.py

# 3. 重新生成报告图表（数据来自 experiment_data.json）
python make_figures.py
```

## 题目接口与输入输出格式

### 题目一 迷宫寻路（`maze_path`）

- 输入：起点 `(row, col)`、终点 `(row, col)`、二维地图（`S` 起点、`E` 终点、`.` 可通过、`X` 障碍）。
- 规则：只能上下左右移动，每移动一格代价为 1。
- 输出：最短路径坐标列表 `[(row, col), ...]`；无解返回 `[]`。
- 启发函数：默认曼哈顿距离，可用 `h_choice` 切换为 `manhattan` / `euclidean` / `zero`。

### 题目二 加权地形（`weighted_cost`）

- 输入：起点、终点、二维地图（数字表示地形权值，`S` / `E` 所在格代价按 0 计）。
- 规则：从节点 A 到相邻节点 B 的代价等于进入 B 的地形权值。
- 输出：最小总代价 `int`；无解返回 `-1`。

### 题目三 极大极小博弈树（`solve_minimax` / `solve_alphabeta`）

- 输入：`TreeNode` 根节点（非叶子有 `children`，叶子有 `value`），根节点为 MAX。
- 规则：MAX 最大化、MIN 最小化，子节点从左到右评估。
- 输出：对 MAX 玩家而言的最佳得分。

### 题目四 竞争环境下的宝藏分配（`TreasureGame`）

- 先由 `distance_matrix()` 用 A* 计算 A、B、T1-T4 两两间最小移动代价（6×6 矩阵）。
- 博弈状态：双方位置、剩余宝藏集合、累计收益（Score_A / Score_B）、当前玩家。
- 行动顺序 A → B → A → B 共四回合，`Gain(p, Ti) = Vi - d(p, Ti)`，`U = Score_A - Score_B`。
- `game.solve(H, use_alphabeta=True)` 返回 `(最优首步, 最优收益差 U, 访问节点数)`，
  U 为整数时按 `int` 返回，与题目预期输出格式一致。

## 测试用例与预期结果

| 题目 | 预期输出 | 本项目实测 |
| --- | --- | --- |
| 题目一 迷宫寻路 | 8 个坐标的最短路径 | 与预期完全一致 |
| 题目一 无解情形 | `[]` | 一致 |
| 题目二 加权地形 | 最小总代价 7 | 一致 |
| 题目二 无解情形 | `-1` | 一致 |
| 题目三 博弈树 | 最佳得分 5 | Minimax 与 Alpha-Beta 均为 5 |
| 题目四 H=2 | `{"best_target": "T2", "best_value": -1}` | 一致 |
| 题目四 H=4 | `{"best_target": "T3", "best_value": 3}` | 一致 |

## 复现与限制

- 所有随机实验使用固定随机种子（`random.seed(42)` 等），多次运行结果一致。
- 性能指标以“扩展节点数 / 访问节点数”为主，辅以单次运行耗时（毫秒级，仅供参考）。
- 已知限制：极小地图与笔直通路上不同启发函数的扩展节点数可能相同，属于正常现象；
  加权地形中启发函数低估越严重，A* 越接近 Dijkstra 的扩展行为。
