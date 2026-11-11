import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figs')

BLUE = '#1f3a5f'
ORANGE = '#c8553d'
GRAY = '#8b94a3'
INK = '#1f2430'
GRID = '#dfe5ee'
FONT = "'Microsoft YaHei','SimHei',sans-serif"
COLORS = [BLUE, ORANGE, GRAY, '#3d6a9e']


def esc(text):
    return (str(text).replace('&', '&amp;').replace('<', '&lt;')
            .replace('>', '&gt;').replace('"', '&quot;'))


def nice_max(value):
    if value <= 0:
        return 1
    mag = 10 ** math.floor(math.log10(value))
    for m in (0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if value <= m * mag:
            return m * mag
    return 10 * mag


def bar_chart(path, title, categories, series, ylabel,
              width=780, height=345):
    left, right, top, bottom = 66, 20, 70, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    ymax = nice_max(max(max(vals) for _, vals in series))
    step = ymax / 5.0

    out = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
               f'height="{height}" viewBox="0 0 {width} {height}" '
               f'font-family="{FONT}">')
    out.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    out.append(f'<text x="{width/2}" y="28" text-anchor="middle" font-size="15" '
               f'font-weight="600" fill="{INK}">{esc(title)}</text>')

    legend_w = sum(18 + len(n) * 13 for n, _ in series)
    lx = (width - legend_w) / 2
    for i, (name, _) in enumerate(series):
        out.append(f'<rect x="{lx:.1f}" y="42" width="12" height="12" fill="{COLORS[i % 4]}"/>')
        out.append(f'<text x="{lx + 17:.1f}" y="52" font-size="11.5" fill="{INK}">{esc(name)}</text>')
        lx += 18 + len(name) * 13

    for i in range(6):
        val = step * i
        y = top + plot_h - plot_h * (val / ymax)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
                   f'stroke="{GRID}" stroke-width="1"/>')
        label = f'{val:g}'
        out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" '
                   f'font-size="11" fill="{GRAY}">{label}</text>')
    out.append(f'<text x="18" y="{top + plot_h / 2:.1f}" transform="rotate(-90 18 '
               f'{top + plot_h / 2:.1f})" text-anchor="middle" font-size="11.5" '
               f'fill="{GRAY}">{esc(ylabel)}</text>')

    out.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
               f'stroke="{BLUE}" stroke-width="1.5"/>')
    out.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
               f'y2="{top + plot_h}" stroke="{BLUE}" stroke-width="1.5"/>')

    group = plot_w / len(categories)
    bar_w = min(30, group * 0.72 / len(series))
    for ci, cat in enumerate(categories):
        gx = left + group * ci
        start = gx + (group - bar_w * len(series)) / 2
        for si, (name, vals) in enumerate(series):
            h = plot_h * (vals[ci] / ymax)
            x = start + si * bar_w
            y = top + plot_h - h
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 3:.1f}" '
                       f'height="{h:.1f}" fill="{COLORS[si % 4]}"/>')
            out.append(f'<text x="{x + (bar_w - 3) / 2:.1f}" y="{y - 5:.1f}" '
                       f'text-anchor="middle" font-size="10.5" fill="{INK}">{vals[ci]}</text>')
        out.append(f'<text x="{gx + group / 2:.1f}" y="{top + plot_h + 20}" '
                   f'text-anchor="middle" font-size="10.5" fill="{INK}">{esc(cat)}</text>')

    out.append('</svg>')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))


def heatmap(path, title, points, matrix, width=460, height=470):
    cell = 52
    left, top = 80, 74
    maxv = max(max(row) for row in matrix) or 1

    def color(v):
        t = v / maxv
        c0 = (240, 244, 250)
        c1 = (31, 58, 95)
        return 'rgb({},{},{})'.format(*[int(c0[i] + (c1[i] - c0[i]) * t) for i in range(3)])

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" font-family="{FONT}">',
           f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
           f'<text x="{width/2}" y="30" text-anchor="middle" font-size="15" '
           f'font-weight="600" fill="{INK}">{esc(title)}</text>']

    for i, p in enumerate(points):
        out.append(f'<text x="{left + cell * i + cell / 2}" y="{top - 10}" '
                   f'text-anchor="middle" font-size="11.5" fill="{INK}">{esc(p)}</text>')
        out.append(f'<text x="{left - 14}" y="{top + cell * i + cell / 2 + 4}" '
                   f'text-anchor="end" font-size="11.5" fill="{INK}">{esc(p)}</text>')
    for i in range(6):
        for j in range(6):
            v = matrix[i][j]
            x = left + cell * j
            y = top + cell * i
            txt = '#ffffff' if v / maxv > 0.55 else INK
            out.append(f'<rect x="{x}" y="{y}" width="{cell - 2}" height="{cell - 2}" '
                       f'fill="{color(v)}" stroke="#ffffff" stroke-width="2"/>')
            out.append(f'<text x="{x + cell / 2 - 1}" y="{y + cell / 2 + 4}" '
                       f'text-anchor="middle" font-size="12" fill="{txt}">{v}</text>')

    bx, by, bw, bh = left, top + cell * 6 + 18, cell * 6 - 26, 10
    for k in range(60):
        t = k / 59
        c = color(t * maxv)
        out.append(f'<rect x="{bx + bw * t:.1f}" y="{by}" width="{bw / 59 + 1:.1f}" '
                   f'height="{bh}" fill="{c}"/>')
    out.append(f'<text x="{bx - 6}" y="{by + 9}" text-anchor="end" font-size="10" '
               f'fill="{GRAY}">0</text>')
    out.append(f'<rect x="{bx + bw + 8}" y="{by - 4}" width="24" height="17" fill="#ffffff"/>')
    out.append(f'<text x="{bx + bw + 12}" y="{by + 9}" font-size="10.5" fill="{INK}">'
               f'{maxv}</text>')
    out.append('</svg>')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    with open(os.path.join(HERE, 'experiment_data.json'), encoding='utf-8') as f:
        data = json.load(f)

    p1 = data['题目一_启发对照']
    cats1 = ['给定迷宫5×5', '随机迷宫30×30 seed=1', '随机迷宫30×30 seed=2']
    dijkstra1 = [r['扩展节点数'] for r in p1 if r['启发函数'] == 'h=0(Dijkstra)']
    man1 = [r['扩展节点数'] for r in p1 if r['启发函数'] == '曼哈顿距离']
    bar_chart(os.path.join(FIG_DIR, 'fig1.svg'), '题目一 不同启发函数的扩展节点数',
              cats1, [('h=0 等价Dijkstra', dijkstra1), ('曼哈顿距离', man1)], '扩展节点数')

    p2 = data['题目二_启发对照']
    cats2 = ['给定5×5', '随机30×30 seed=3', '随机30×30 seed=4', '低代价30×30 seed=5']
    zero2 = [r['扩展节点数'] for r in p2 if r['启发函数'] == 'h=0(Dijkstra)']
    man2 = [r['扩展节点数'] for r in p2 if r['启发函数'] == '曼哈顿距离']
    euc2 = [r['扩展节点数'] for r in p2 if r['启发函数'] == '欧氏距离']
    bar_chart(os.path.join(FIG_DIR, 'fig2.svg'), '题目二 三种启发函数的扩展节点数',
              cats2, [('h=0 等价Dijkstra', zero2), ('曼哈顿距离', man2), ('欧氏距离', euc2)],
              '扩展节点数')

    p3 = data['题目三_剪枝对照']
    cats3 = ['给定树', '随机树 深度3分支4', '随机树 深度4分支3']
    mm3 = [r['访问节点数'] for r in p3 if r['算法'] == 'Minimax']
    ab3 = [r['访问节点数'] for r in p3 if r['算法'] == 'Alpha-Beta']
    bar_chart(os.path.join(FIG_DIR, 'fig3.svg'), '题目三 Minimax与Alpha-Beta访问节点数',
              cats3, [('Minimax', mm3), ('Alpha-Beta', ab3)], '访问节点数')

    p4 = data['题目三_访问顺序']
    cats4 = ['随机树 深度3分支4', '随机树 深度4分支3']
    left_order = [r['访问节点数'] for r in p4 if r['访问顺序'] == '从左到右']
    best_order = [r['访问节点数'] for r in p4 if r['访问顺序'] == '最佳排序']
    bar_chart(os.path.join(FIG_DIR, 'fig4.svg'), '题目三 访问顺序对Alpha-Beta剪枝的影响',
              cats4, [('从左到右', left_order), ('最佳排序', best_order)], '访问节点数')

    d4 = data['题目四_距离矩阵']
    heatmap(os.path.join(FIG_DIR, 'fig5.svg'), '题目四 6×6最小移动代价矩阵',
            d4['points'], d4['matrix'])

    p5 = data['题目四_深度与剪枝对照']
    t5 = data['题目四_访问顺序']
    cats6 = ['H=2', 'H=4']
    mm6 = [r['访问节点数'] for r in p5 if r['算法'] == 'Minimax']
    ab6 = [r['访问节点数'] for r in p5 if r['算法'] == 'Alpha-Beta']
    vab6 = [r['访问节点数'] for r in t5 if r['访问顺序'] == '按价值降序']
    bar_chart(os.path.join(FIG_DIR, 'fig6.svg'), '题目四 搜索开销与访问顺序',
              cats6, [('Minimax', mm6), ('Alpha-Beta 字典序', ab6),
                      ('Alpha-Beta 价值降序', vab6)], '访问节点数')

    print('已生成图表：', ', '.join(sorted(os.listdir(FIG_DIR))))


if __name__ == '__main__':
    main()
