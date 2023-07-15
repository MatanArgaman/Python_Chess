import numpy

from algorithms.mcts import basic_evaluation, Node


def alpha_beta(board, depth, node):
    if depth == 0:
        # if board.turn:
        #     v = -np.inf
        #     v = max(v, capturing_moves(board, node, 0))
        #     node.alpha = max(node.alpha, v)
        # else:
        #     v = np.inf
        #     v = min(v, capturing_moves(board, node, 0))
        #     node.beta = min(node.alpha, v)
        v = basic_evaluation(node.board)
        if not board.turn:
            node.alpha = v
        else:
            node.beta = v
        return v
    if board.turn:
        v = -np.inf
        for move in node.board.legal_moves:
            child_node = Node(board, node.alpha, node.beta, move=move)
            node.child_nodes.append(child_node)
            v = max(v, alpha_beta(child_node.board, depth - 1, child_node))
            node.alpha = max(node.alpha, v)
            if node.alpha >= node.beta:
                break
    else:
        v = np.inf
        for move in node.board.legal_moves:
            child_node = Node(board, node.alpha, node.beta, move=move)
            node.child_nodes.append(child_node)
            v = min(v, alpha_beta(child_node.board, depth - 1, child_node))
            node.beta = min(node.beta, v)
            if node.alpha >= node.beta:
                break
    return v


def alpha_beta_move(board):
    # import cProfile, pstats, io
    # pr = cProfile.Profile()
    # pr.enable()

    max_depth = 4
    # min_nodes = 15000
    Node.reset_counter()
    # while Node.counter<min_nodes:
    Node.reset_counter()
    root = Node(board, -np.inf, np.inf)
    v = alpha_beta(board, max_depth, root)  # must be an even number to end turn in opponent's turn.
    # max_depth+=2

    root.child_nodes.sort(key=lambda x: -x.beta if board.turn else x.alpha)
    print('total nodes explored:', Node.counter, v, max_depth)
    # equivalent_moves = [root.child_nodes[0]]
    # for i in range(1, len(root.child_nodes)):
    #     if board.turn:
    #         if root.child_nodes[0].beta == root.child_nodes[i].beta:
    #             equivalent_moves.append(root.child_nodes[i])
    #     else:
    #         if root.child_nodes[0].alpha == root.child_nodes[i].alpha:
    #             equivalent_moves.append(root.child_nodes[i])
    # return equivalent_moves[np.random.randint(0, len(equivalent_moves))].move

    # pr.disable()
    # s = io.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())

    return root.child_nodes[0].move


def capturing_moves(board, node, depth):
    if depth >= 4:
        v = basic_evaluation(node.board)
        if not board.turn:
            node.alpha = v
        else:
            node.beta = v
        return v
    captured_move_available = False
    if board.turn:
        v = -np.inf
        for move in node.board.legal_moves:
            if board.is_capture(move):
                captured_move_available = True
                child_node = Node(board, node.alpha, node.beta, move=move)
                node.child_nodes.append(child_node)
                v = max(v, capturing_moves(child_node.board, child_node, depth + 1))
                node.alpha = max(node.alpha, v)
                if node.alpha >= node.beta:
                    break
    else:
        v = np.inf
        for move in node.board.legal_moves:
            if board.is_capture(move):
                captured_move_available = True
                child_node = Node(board, node.alpha, node.beta, move=move)
                node.child_nodes.append(child_node)
                v = min(v, capturing_moves(child_node.board, child_node, depth + 1))
                node.beta = min(node.beta, v)
                if node.alpha >= node.beta:
                    break
    if not captured_move_available:
        v = basic_evaluation(node.board)
        if not board.turn:
            node.alpha = v
        else:
            node.beta = v
    return v
