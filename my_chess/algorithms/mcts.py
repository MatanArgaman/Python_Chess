import heapq
import random
import time
from multiprocessing import Pool

import chess
import numpy as np
import tqdm
from typing import List, Optional
import torch

from predict import get_input_representation
from shared.shared_functionality import get_nn_moves_and_probabilities, load_pytorch_model, SingletonLogger, \
    load_tensorflow_model
from nn.pytorch_nn.AlphaChess.utils import create_alpha_chess_model, load_model

LOG = SingletonLogger().get_logger('play')


def get_definitive_value(board: chess.Board) -> Optional[float]:
    if board.is_checkmate():
        return 0
    if board.is_insufficient_material() or board.is_stalemate():
        return 0.5
    return None


class MCTS_Node:
    counter: int = 0
    use_nn: bool = False

    def __init__(self, board, move=None, depth=0, parent_node=None, is_calc_all=False,
                 nn_model=None, device: str = None, is_torch_nn: bool = False):
        MCTS_Node.counter += 1
        self.won: int = 0
        self.played: int = 0
        self.parent_node: MCTS_Node = parent_node
        self.move: chess.Move = move
        self.board: chess.Board = board.copy()
        self.depth: int = depth
        self.nn_model = nn_model
        self.device: str = device
        self.is_torch_nn: bool = is_torch_nn
        self.counter: int = MCTS_Node.counter
        if move is not None:
            self.board.push(self.move)
        self.child_nodes: List[MCTS_Node] = []
        self.legal_moves: set = set(self.board.legal_moves)

        if is_calc_all:
            # calculate capturing moves
            self.capturing_moves: List[chess.Move] = []
            start_pos_dict: dict = {}

            for i, m in enumerate(self.legal_moves):
                move_str = str(m)
                start_pos = move_str[:2]
                start_pos_dict[start_pos] = start_pos_dict.get(start_pos, []) + [m]
                if self.board.is_capture(m):
                    self.capturing_moves.append(m)

            # calculate avoidance moves
            self.avoidance_moves: List[chess.Move] = []

            self.board.turn = not self.board.turn
            # self.board.legal_moves here is not identical to self.legal_moves as the turn was changed.
            for i, m in enumerate(self.board.legal_moves):
                if self.board.is_capture(m):
                    start_pos = str(m)[2:4]
                    self.avoidance_moves += start_pos_dict.get(start_pos, [])
            self.board.turn = not self.board.turn
            self.avoidance_moves = list(set(self.avoidance_moves))

        # calculate nn best moves
        self.best_moves: List[chess.Move] = []

        if MCTS_Node.use_nn is not None:
            self.get_nn_moves()

        if len(self.legal_moves) <= 1:
            self.best_moves = list(self.legal_moves)
        if not self.best_moves:
            self.best_moves += [random.choice(list(self.legal_moves))]
            LOG.warning("used a random move as the nn didn't provide a legal one")

        self.ordered_unexplored_moves: List[chess.Move] = self.best_moves
        if is_calc_all:
            # create a list of all moves to be considered in the mcts
            best_moves_set = set(self.best_moves)
            self.ordered_unexplored_moves += list(set(self.avoidance_moves).difference(best_moves_set))
            self.ordered_unexplored_moves += list(set(self.capturing_moves).difference(best_moves_set))
            additional_moves = list(self.legal_moves.difference(self.ordered_unexplored_moves))
            if additional_moves:
                self.ordered_unexplored_moves += [random.choice(additional_moves)]

        if not self.best_moves:  # heuristic which shouldn't occur. todo: check how can all nn_moves not be legal.
            if not self.board.is_game_over():
                if is_calc_all and self.ordered_unexplored_moves:
                    self.best_moves = [self.ordered_unexplored_moves[0]]
                else:
                    self.best_moves = [list(self.legal_moves)[0]]

    def get_nn_moves(self, k_best_moves=3):
        nn_moves, _, nn_values = get_nn_moves_and_probabilities([self.board], self.nn_model, k_best_moves=15,
                                                                is_torch_nn=self.is_torch_nn,
                                                                device=self.device)
        self.value: float = (nn_values.item() + 1) / 2.0  # move from [-1, 1] range to [0, 1] range
        definitive_value: Optional[float] = get_definitive_value(self.board)
        if definitive_value is not None:
            self.value = definitive_value

        added_moves = 0
        for m in nn_moves[0]:
            try:
                m = chess.Move.from_uci(m)
            except:  # may fail due to illegal move
                continue
            if m in self.legal_moves:
                self.best_moves.append(m)
                added_moves += 1
            if added_moves >= k_best_moves:
                break

    def select(self) -> 'MCTS_Node':
        if self.ordered_unexplored_moves:
            return self.expand()

        # if game is either definitely a stalemate or a checkmate,
        # return this Node so that its value will be used as the reward.
        definitive_value = get_definitive_value(self.board)
        if definitive_value is not None:
            return self

        node: MCTS_Node = self.get_best_child()
        return node.select()

    def get_best_child(self) -> 'MCTS_Node':
        best_value: float = float("-inf")
        best_nodes: List[MCTS_Node] = []
        for node in self.child_nodes:
            ams = node.calc_AMS()
            if ams > best_value:
                best_nodes = [node]
                best_value = ams
            elif ams == best_value:
                best_nodes.append(node)
        return random.choice(best_nodes)

    def expand(self) -> 'MCTS_Node':
        return self.add_new_child(self.ordered_unexplored_moves.pop(0))

    @staticmethod
    def reset_counter() -> None:
        MCTS_Node.counter = 0

    def win_percentage(self) -> float:
        if self.played > 0:
            return float(self.won) / self.played
        raise Exception("No game played")

    def calc_AMS(self) -> float:
        if self.played <= 0:
            return float('inf')
        if self.parent_node is None:
            return 1
        exploitation = self.win_percentage()
        exploration = np.sqrt(2 * np.log(self.parent_node.played) / self.played)
        return exploitation + exploration

    def add_new_child(self, move: chess.Move) -> 'MCTS_Node':
        assert move in self.legal_moves
        node = self.create_new_child(move)
        self.child_nodes.append(node)
        return node

    def create_new_child(self, move: chess.Move, is_calc_all: bool = False) -> 'MCTS_Node':
        node = MCTS_Node(self.board, move=move, depth=self.depth + 1,
                         parent_node=self, is_calc_all=is_calc_all,
                         nn_model=self.nn_model, device=self.device,
                         is_torch_nn=self.is_torch_nn)
        return node

    def get_nn_best_move(self) -> chess.Move:
        return self.best_moves[0]


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


class Node:
    counter = 0

    def __init__(self, board, alpha, beta, parent, move=None):
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
        self.parent = parent

    @staticmethod
    def reset_counter():
        Node.counter = 0


def merge_trees(node1, node2):
    node_map = dict([(n.move, n) for n in node1.child_nodes])
    for n in node2.child_nodes:
        if n.move in node_map:
            node_map[n.move].played += n.played
            node_map[n.move].won += n.won
            # merge_trees(node_map[n.move], n)
        else:
            move_index = node1.legal_moves.index(n.move)
            node1.child_nodes.append(n)
            node1.explored_moves.add(move_index)
            n.parent_node = node1
            n.update_parent_heap()


# # debugging script to show game moves according to tree
# b = board.copy()
# for m in game_moves:
#     print(str(m))
#     print(b)
#     print('\n')
#     b.push(m)

# mcts_process_num = multiprocessing.cpu_count() - 1


def mcts_move(board, nn_model, device, max_games=800, k_best_moves=2):
    root = mcts_move_helper(board, max_games, nn_model, device)
    best_nodes = [(n, n.win_percentage()) for n in root.child_nodes]
    sorted_nodes = sorted(best_nodes, key=lambda x: x[1])
    for item in sorted_nodes:
        LOG.debug(item[0].move, item[1], item[0].played)
    k_best_nodes = sorted_nodes[-k_best_moves:][::-1]
    moves = [n[0].move for n in k_best_nodes]
    return moves


def get_nn_and_device(is_torch_nn, config):
    if is_torch_nn:
        import torch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = create_alpha_chess_model(device, True, True, True)
        load_model(model, config['play']['torch_nn_path'], config)
        return device, model
    else:
        return load_tensorflow_model(config)


def mcts_move_helper(board, max_games, nn_model, device) -> MCTS_Node:
    start_time = time.time()  # about 24 seconds for a single processor

    MCTS_Node.reset_counter()
    root: MCTS_Node = MCTS_Node(board, nn_model=nn_model, device=device, is_torch_nn=True)

    print('computer calculating move')
    for _ in tqdm.tqdm(range(max_games)):
        # Selection + Expansion
        selected_node: MCTS_Node = root.select()

        # backpropagation
        reward = 1 - selected_node.value
        node = selected_node
        while node is not None:
            node.played += 1
            node.won += reward
            node = node.parent_node
            reward = 1.0 - reward

    LOG.debug('total move time:', time.time() - start_time)
    return root


def visualize_tree(node, depth=np.inf):
    from graphviz import Digraph
    dot = Digraph(comment='Graph', format='png')
    dot.graph_attr['bgcolor'] = 'cyan'
    dot.node_attr.update(style='filled', fontsize='15', fixedsize='false', fontcolor='blue')
    edges = []

    def node_to_graph_node(node, dot):
        dot.node(str(node.counter),
                 f'{round(node.won, 2)}/{node.played}\nm:{node.move}\n v:{round(node.value, 3)}\n'
                 f'{node.counter}\nams:{round(node.calc_AMS(), 2)}',
                 shape='box' if node.board.turn else 'oval', color='white' if node.board.turn else 'black')

    def helper(node, edges, dot, depth):
        if depth <= 0:
            return
        for n in node.child_nodes:
            node_to_graph_node(n, dot)
            edges.append((str(node.counter), str(n.counter)))
            helper(n, edges, dot, depth - 1)

    node_to_graph_node(node, dot)
    helper(node, edges, dot, depth)
    dot.edges(edges)
    print(dot.source)
    dot.render('test-output/round-table.gv', view=True)


def is_game_over(board):
    if board.is_checkmate() or board.is_insufficient_material() or board.is_stalemate():
        return True
    return False


def get_material_reward(board, start_turn_is_white, start_material_score):
    material_score = get_material_score(board) - start_material_score

    if start_turn_is_white:
        if material_score > 0:
            reward = 0.6
        elif material_score < 0:
            reward = 0.4
        else:
            reward = 0.5
    else:
        if material_score > 0:
            reward = 0.4
        elif material_score < 0:
            reward = 0.6
        else:
            reward = 0.5
    return reward


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
