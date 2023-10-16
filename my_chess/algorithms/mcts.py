import random
import time
import chess
import numpy as np
import tqdm
import torch
from typing import List, Optional

from shared.shared_functionality import get_nn_moves_and_probabilities, SingletonLogger
from nn.pytorch_nn.AlphaChess.utils import create_alpha_chess_model, load_model

LOG = SingletonLogger().get_logger('play')


def get_definitive_value(board: chess.Board) -> Optional[float]:
    if board.is_checkmate():
        return 0
    if board.is_insufficient_material() or board.is_stalemate() or board.is_repetition():
        return 0.5
    return None


class MCTS_Node:
    counter: int = 0

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

        # calculate nn best moves
        self.best_moves: List[chess.Move] = []

        self.get_nn_moves()

        if len(self.legal_moves) <= 1:
            self.best_moves = list(self.legal_moves)
        if not self.best_moves and self.legal_moves:
            self.best_moves += [random.choice(list(self.legal_moves))]
            LOG.warning("used a random move as the nn didn't provide a legal one")

        self.ordered_unexplored_moves: List[chess.Move] = self.best_moves

        if not self.best_moves:  # heuristic which shouldn't occur. todo: check how can all nn_moves not be legal.
            if not self.board.is_game_over():
                if is_calc_all and self.ordered_unexplored_moves:
                    self.best_moves = [self.ordered_unexplored_moves[0]]
                else:
                    self.best_moves = [list(self.legal_moves)[0]]

    def get_nn_moves(self, k_best_moves=5):
        nn_moves, _, nn_values = get_nn_moves_and_probabilities([self.board], self.nn_model, k_best_moves=15,
                                                                is_torch_nn=self.is_torch_nn,
                                                                device=self.device)
        self.value: float = (nn_values.item() + 1) / 2.0  # move from [-1, 1] range to [0, 1] range
        definitive_value: Optional[float] = get_definitive_value(self.board)
        if definitive_value is not None:
            self.value = definitive_value

        added_moves = 0
        for m in nn_moves[0]:
            m1 = m2 = None
            try:
                m1 = chess.Move.from_uci(m)
            except chess.InvalidMoveError:  # may fail due to illegal move
                pass
            try:
                m2 = chess.Move.from_uci(m + 'q')
            except chess.InvalidMoveError:  # may fail due to illegal move
                pass
            if m1 or m2:
                if m1 in self.legal_moves:
                    self.best_moves.append(m1)
                    added_moves += 1
                elif m2 in self.legal_moves:
                    self.best_moves.append(m2)
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

def mcts_move(board, nn_model, device, max_games=1200, k_best_moves=2):
    root = mcts_move_helper(board, max_games, nn_model, device)
    best_nodes = [(n, n.win_percentage()) for n in root.child_nodes]
    sorted_nodes = sorted(best_nodes, key=lambda x: x[1])
    for item in sorted_nodes:
        LOG.debug(item[0].move, item[1], item[0].played)
    k_best_nodes = sorted_nodes[-k_best_moves:][::-1]
    moves = [n[0].move for n in k_best_nodes]
    return moves


def get_nn_and_device(config):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = create_alpha_chess_model(device, True, True, True)
        load_model(model, config['play']['torch_nn_path'], config)
        return device, model


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


def visualize_tree(node, depth=np.inf, is_flipped=False):
    from graphviz import Digraph
    from shared.shared_functionality import move_to_mirror_move
    dot = Digraph(comment='Graph', format='png')
    dot.graph_attr['bgcolor'] = 'cyan'
    dot.node_attr.update(style='filled', fontsize='15', fixedsize='false', fontcolor='blue')
    edges = []

    def node_to_graph_node(node, dot):
        move = node.move
        if is_flipped and move:
            move = move_to_mirror_move(str(node.move), flip_horizontally=True)
        dot.node(str(node.counter),
                 f'{round(node.won, 2)}/{node.played}({round((node.won / node.played), 2)})\nm:{move}\n v:{round(node.value, 3)}\n'
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