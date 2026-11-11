class TreeNode:

    def __init__(self, value=None, children=None):
        self.value = value
        self.children = children if children is not None else []


def _is_leaf(node):
    return len(node.children) == 0


def minimax(node, is_max=True, stats=None):
    if stats is None:
        stats = {}
    stats['visited'] = stats.get('visited', 0) + 1

    if _is_leaf(node):
        return node.value

    if is_max:
        best = float('-inf')
        for child in node.children:
            v = minimax(child, False, stats)
            best = max(best, v)
        return best
    else:
        best = float('inf')
        for child in node.children:
            v = minimax(child, True, stats)
            best = min(best, v)
        return best


def alphabeta(node, alpha, beta, is_max=True, stats=None):
    if stats is None:
        stats = {}
    stats['visited'] = stats.get('visited', 0) + 1

    if _is_leaf(node):
        return node.value

    if is_max:
        best = float('-inf')
        for child in node.children:
            v = alphabeta(child, alpha, beta, False, stats)
            best = max(best, v)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    else:
        best = float('inf')
        for child in node.children:
            v = alphabeta(child, alpha, beta, True, stats)
            best = min(best, v)
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best


def solve_minimax(root):
    stats = {}
    value = minimax(root, True, stats)
    return value, stats['visited']


def solve_alphabeta(root):
    stats = {}
    value = alphabeta(root, float('-inf'), float('inf'), True, stats)
    return value, stats['visited']


def order_children(root, key):
    new_node = TreeNode(value=root.value, children=[])
    for child in root.children:
        new_node.children.append(order_children(child, key))
    new_node.children.sort(key=key)
    return new_node
