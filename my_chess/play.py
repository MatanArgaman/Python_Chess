import chess.svg
import numpy as np
import time
import argparse
import heapq
import os
import pickle
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import QtCore
import json
import tqdm
from multiprocessing import Pool
import multiprocessing

from shared.shared_functionality import *


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setGeometry(100, 100, 1100, 1100)

        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(10, 10, 1080, 1080)

        if args.board is not None:
            self.chessboard = chess.Board(args.board)
        else:
            self.chessboard = chess.Board()

        self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg)
        self.widgetSvg.mousePressEvent = self.onclick
        self.human_first_click = True
        self.human_move = ''

        with open(os.path.join(os.getcwd(), 'config.json'), 'r') as f:
            self.config = json.load(f)

        self.nn_model = None
        self.use_nn = args.nn
        self.use_database = args.database
        self.use_mcts = args.mcts

        # QTimer.singleShot(10, self.play)
        # QTimer.singleShot(10, self.play)

    def human_on_click(self, event):

        move = self.get_human_move(event)
        if move is None:
            return None
        if self.human_first_click:
            self.human_first_click = False
            self.human_move = move
        else:
            self.human_first_click = True
            self.human_move += move
            try:
                move = None
                start_position = position_to_index_1d(self.human_move[:2])
                if self.chessboard.piece_at(start_position) in [chess.Piece.from_symbol('p'),
                                                                chess.Piece.from_symbol('P')]:
                    end_position = position_to_index_1d(self.human_move[2:])
                    if start_position in [end_position + ROW_SIZE, end_position - ROW_SIZE]:
                        print('here')
                        value = input()
                        if value in ['q', 'r', 'b', 'n']:
                            move = chess.Move.from_uci(self.human_move + value)
                if move is None:
                    move = chess.Move.from_uci(self.human_move)
                if move in self.chessboard.legal_moves:
                    # check for pawn to last row move and prompt player for type of piece conversion wanted
                    return move
            except:
                pass
        return None

    def get_computer_move(self):
        if self.use_database:
            try:
                con_db = self.config['database']
                database = get_database_from_file(self.chessboard.fen(), con_db['file_path'], con_db['file_name'])
                moves, probabilities = get_fen_moves_and_probabilities(database, self.chessboard.fen())
                index = np.searchsorted(probabilities.cumsum(), np.random.rand(), side='left')
                return chess.Move.from_uci(moves[index])
            except:
                pass
        if self.use_mcts:
            move = mcts_move(self.chessboard)[0]
            return move
        if self.use_nn:
            try:
                if self.nn_model is None:
                    from tensorflow import keras
                    self.nn_model = keras.models.load_model(self.config['train']['nn_model_path'])
                moves = get_nn_moves([self.chessboard.copy()], self.nn_model)[0]  # returns the best k moves
                for m in moves:
                    if chess.Move.from_uci(m) in self.chessboard.legal_moves:
                        return chess.Move.from_uci(m)
            except:
                pass
        # if no legal move was generated use alpha beta to find one.
        return alpha_beta_move(self.chessboard)

    def onclick(self, event):

        if event.button() == QtCore.Qt.LeftButton:
            if self.chessboard.turn:
                if args.whuman:
                    move = self.human_on_click(event)
                    if move is None:
                        return
                else:
                    move = self.get_computer_move()
                    self.human_first_click = True
                print('white:', str(move))
            else:
                if args.bhuman:
                    move = self.human_on_click(event)
                    if move is None:
                        return
                else:
                    move = self.get_computer_move()
                    self.human_first_click = True
                print('black:', str(move))

            self.chessboard.push(move)
            self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
            self.widgetSvg.load(self.chessboardSvg)
            if self.chessboard.is_checkmate():
                if self.chessboard.turn:
                    print('Black Wins')
                else:
                    print('White Wins')
            if self.chessboard.is_insufficient_material():
                print('Draw - insufficient material')
            if self.chessboard.is_stalemate():
                print('Draw - stalemate')

        if event.button() == QtCore.Qt.RightButton:  # undo last move
            self.chessboard.pop()
            self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
            self.widgetSvg.load(self.chessboardSvg)
        if event.button() == QtCore.Qt.MiddleButton:
            print(self.chessboard.__repr__())

        # self.widgetSvg.update()
        # time.sleep(1)

    def get_human_move(self, event):
        SQUARE_START = 40
        SQUARE_SIZE = 125
        SQUARES_PER_ROW_COLUMN = 8

        def get_square_index(pos):
            v = (pos - SQUARE_START) // SQUARE_SIZE
            if 0 <= v < SQUARES_PER_ROW_COLUMN:
                return v
            return None

        row = get_square_index(event.x())
        if row is None:
            return None
        row = chr(ord('a') + row)
        col = get_square_index(event.y())
        if col is None:
            return None
        col = SQUARES_PER_ROW_COLUMN - col
        return str(row) + str(col)

    def play(self):
        self.widgetSvg.update()
        self.show()
        time.sleep(1)
        for i in range(3):
            move = alpha_beta_move(self.chessboard)
            self.chessboard.push(move)
            self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
            self.widgetSvg.load(self.chessboardSvg)
            self.widgetSvg.update()
            time.sleep(1)


class Node:
    counter = 0

    def __init__(self, board, alpha, beta, move=None):
        Node.counter += 1
        if Node.counter % 1000 == 0:
            print(Node.counter)
        self.alpha = alpha
        self.beta = beta
        self.move = move
        self.board = board.copy()
        if move is not None:
            self.board.push(self.move)
        self.child_nodes = []

    @staticmethod
    def reset_counter():
        Node.counter = 0


class MCTS_Node:
    counter = 0

    def __init__(self, board, parent_node=None, move=None, depth=0):
        MCTS_Node.counter += 1
        self.won = 0
        self.played = 0
        self.parent_node = parent_node
        self.move = move
        self.capturing_move =False
        if move is not None:
            if board.is_capture(move):
                self.capturing_move = True
        self.board = board.copy()
        self.explored_moves = set()
        self.child_ams_heapq = []
        self.depth = depth
        if move is not None:
            self.board.push(self.move)
        self.legal_moves = [m for m in self.board.legal_moves]
        self.child_nodes = []

    @staticmethod
    def reset_counter():
        MCTS_Node.counter = 0

    def win_percentage(self):
        if self.played > 0:
            return float(self.won) / self.played
        raise Exception("No game played")

    def calc_AMS(self):
        assert self.played > 0
        if self.parent_node is None:
            return 1
        exploitation = self.win_percentage()
        exploration = np.log(self.parent_node.played) / self.played
        capturing_heuristic = (0.5 if self.capturing_move else 0) / self.played
        return exploitation + np.sqrt(2 * exploration + capturing_heuristic)

    def add_new_child(self, move, move_index):
        assert self.legal_moves[move_index] == move
        node = MCTS_Node(self.board, self, move, self.depth + 1)
        self.child_nodes.append(node)
        self.explored_moves.add(move_index)
        return node

    def get_child_node(self):
        '''
        :return: a child node according to adaptive multi stage sampling
        '''
        NEW_NODE_CHANCE = 0.2
        new_move = False
        if self.child_nodes:
            if np.random.rand() <= NEW_NODE_CHANCE:
                new_move = True
        else:
            new_move = True
        if len(self.legal_moves) == len(self.explored_moves):  # all possible moves where explored
            new_move = False
        if new_move:
            CAPTURING_MOVES_CHANCE = 0.5
            available_capturing_moves = []
            for i, m in enumerate(self.legal_moves):
                if (i not in self.explored_moves) and self.board.is_capture(m):
                    available_capturing_moves.append(i)

            if available_capturing_moves and np.random.rand() <= CAPTURING_MOVES_CHANCE:
                unexplored_moves = set(available_capturing_moves)
            else:
                unexplored_moves = set(np.arange(len(self.legal_moves))).difference(self.explored_moves)
            move_index = list(unexplored_moves)[np.random.randint(len(unexplored_moves))]
            move = self.legal_moves[move_index]
            child_node = self.add_new_child(move, move_index)
        else:
            child_node = heapq.heappop(self.child_ams_heapq)[2]
        return child_node

    def update_parent_heap(self):
        heapq.heappush(self.parent_node.child_ams_heapq, (-self.calc_AMS(), id(self), self))


def get_material_score(board):
    wp = len(board.pieces(chess.PAWN, chess.WHITE))
    wr = len(board.pieces(chess.ROOK, chess.WHITE))
    wb = len(board.pieces(chess.BISHOP, chess.WHITE))
    wk = len(board.pieces(chess.KNIGHT, chess.WHITE))
    wq = len(board.pieces(chess.QUEEN, chess.WHITE))

    bp = len(board.pieces(chess.PAWN, chess.BLACK))
    br = len(board.pieces(chess.ROOK, chess.BLACK))
    bb = len(board.pieces(chess.BISHOP, chess.BLACK))
    bk = len(board.pieces(chess.KNIGHT, chess.BLACK))
    bq = len(board.pieces(chess.QUEEN, chess.BLACK))

    material_score = (wp - bp) + (wr - br) * 5 + (wb - bb) * 4 + (wk - bk) * 3 + (wq - bq) * 9
    return material_score


def basic_evaluation(board):
    if board.is_checkmate():
        if board.turn:
            return np.inf
        else:
            return -np.inf
    if board.is_insufficient_material():
        return 0
    if board.is_stalemate():
        return 0

    score = get_material_score(board)
    return score


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


def is_game_over(board):
    if board.is_checkmate() or board.is_insufficient_material() or board.is_stalemate():
        return True
    return False

def merge_trees(node1, node2):
    node_map = dict([(n.move, n) for n in node1.child_nodes])
    for n in node2.child_nodes:
        if n.move in node_map:
            node_map[n.move].played += n.played
            node_map[n.move].won += n.won
            merge_trees(node_map[n.move], n)
        else:
            move_index = node1.legal_moves.index(n.move)
            node1.child_nodes.append(n)
            node1.explored_moves.add(move_index)
            n.parent_node = node1
            n.update_parent_heap()

def mcts_move(board, max_games=550, max_depth=20, k_best_moves=5):
    process_num = multiprocessing.cpu_count() - 1
    indices = [(board, max_games, max_depth, k_best_moves)] * (process_num)
    first_root = None
    with Pool(process_num) as p:
        for root in tqdm.tqdm(p.imap(mcts_move_helper, indices), total=len(indices)):
            if first_root is None:
                first_root = root
            else:
                merge_trees(first_root, root)

    root = first_root
    best_nodes = [(n, n.win_percentage()) for n in root.child_nodes]
    sorted_nodes = sorted(best_nodes, key=lambda x: x[1])
    k_best_nodes = sorted_nodes[-k_best_moves:][::-1]
    moves = [n[0].move for n in k_best_nodes]
    return moves


def mcts_move_helper(parameters):
    board, max_games, max_depth, k_best_moves = parameters
    start_time = time.time()  # about 24 seconds for a single processor

    def get_material_reward(board, start_turn_is_white):
        material_score = get_material_score(board)
        if material_score > 0:
            reward = 1
        elif material_score < 0:
            reward = 0
        else:
            reward = 0.5
        if not start_turn_is_white:
            reward = 1 - reward
        return reward

    MCTS_Node.reset_counter()
    root = MCTS_Node(board)

    while root.played < max_games:
        node = root
        game_nodes = [root]
        while (not node.board.is_game_over()) and node.depth < max_depth:
            node = node.get_child_node()
            game_nodes.append(node)

        if node.board.is_insufficient_material() or node.board.is_stalemate():
            reward = 0.5
        else:
            if board.turn:
                if not node.board.turn:
                    if board.is_checkmate:
                        reward = 1
                    else:
                        if node.depth == max_depth:
                            reward = 0.5
                        else:
                            reward = get_material_reward(node.board, board.turn)
                else:
                    if board.is_checkmate:
                        reward = 0
                    else:
                        if node.depth == max_depth:
                            reward = 0.5
                        else:
                            reward = get_material_reward(node.board, board.turn)
            else:
                if not node.board.turn:
                    if board.is_checkmate:
                        reward = 0
                    else:
                        if node.depth == max_depth:
                            reward = 0.5
                        else:
                            reward = get_material_reward(node.board, board.turn)
                else:
                    if board.is_checkmate:
                        reward = 1
                    else:
                        if node.depth == max_depth:
                            reward = 0.5
                        else:
                            reward = get_material_reward(node.board, board.turn)

        for node in game_nodes:
            node.played += 1
            node.won += reward
            if node.parent_node:
                node.update_parent_heap()

    print('total move time:', time.time() - start_time)
    return root


def get_nn_moves(board_list, model, k_best_moves=5):
    from predict import get_input_representation, output_representation_to_moves_and_probabilities, \
        sort_moves_and_probabilities
    input_representation = np.zeros([len(board_list), 8, 8, INPUT_PLANES])
    for i, board in enumerate(board_list):
        board_turn = board.turn
        if not board_turn:
            board = board.mirror()
        input_representation[i] = get_input_representation(board, 0)[np.newaxis]
    output = model.predict(input_representation)
    res = []
    for i in range(output.shape[0]):

        o = output[i]
        threshold = np.sort(o.flatten())[-k_best_moves]
        a, b, c = np.where(o >= threshold)
        o2 = np.zeros([8, 8, OUTPUT_PLANES])
        o2[a, b, c] = o[a, b, c]
        m, p = output_representation_to_moves_and_probabilities(o2)
        if m.size == 0:
            res.append([])
            continue
        m, p = sort_moves_and_probabilities(m, p)
        if not board_list[i].turn:
            m = [move_to_mirror_move(move) for move in m]
        res.append(m)
    return res


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


def visualize_tree(node, depth=np.inf):
    from graphviz import Digraph
    dot = Digraph(comment='Alpha Beta graph', format='png')
    dot.node_attr.update(style='filled', fontsize='15', fixedsize='false', fontcolor='blue')
    edges = []
    node_counter = [0]

    def node_to_graph_node(node, dot):
        dot.node(str(node.counter), 'alpha:{0} beta:{1}, move:{2}'.format(node.alpha, node.beta, str(node.move)),
                 shape='box' if node.board.turn else 'oval', color='black' if node.board.turn else 'white')

    def helper(node, node_counter, edges, dot, depth):
        if depth <= 0:
            return
        for n in node.child_nodes:
            n.counter = node_counter[0]
            node_counter[0] += 1
            node_to_graph_node(n, dot)
            edges.append((str(node.counter), str(n.counter)))
            helper(n, node_counter, edges, dot, depth - 1)

    node.counter = node_counter[0]
    node_to_graph_node(node, dot)
    node_counter[0] += 1
    helper(node, node_counter, edges, dot, depth)
    dot.edges(edges)
    print(dot.source)
    dot.render('test-output/round-table.gv', view=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bhuman', action='store_true')
    parser.add_argument('--whuman', action='store_true')
    parser.add_argument('--database', action='store_true', help='get moves from database if available')
    parser.add_argument('--nn', action='store_true', help='get moves from neural network predictions')
    parser.add_argument('--mcts', action='store_true', help='get moves from monte carlo tree search')
    parser.add_argument('-board', help='start from predefined board (fen), e.g: "8/4Q3/8/7k/5K2/8/8/8 w - - 0 1"')

    args = parser.parse_args()
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
