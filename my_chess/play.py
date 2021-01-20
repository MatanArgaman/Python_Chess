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

from shared.shared_functionality import *

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setGeometry(100, 100, 1100, 1100)

        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(10, 10, 1080, 1080)

        self.chessboard = chess.Board()

        self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg)
        self.widgetSvg.mousePressEvent = self.onclick
        self.human_first_click=True
        self.human_move=''
        self.database_path = args.memory
        # QTimer.singleShot(10, self.play)
        # QTimer.singleShot(10, self.play)


    def human_on_click(self,event):

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
                        value = input()
                        if value in ['q', 'r', 'b', 'n']:
                            move =  chess.Move.from_uci(self.human_move + value)
                if move is None:
                    move = chess.Move.from_uci(self.human_move)
                if move in self.chessboard.legal_moves:
                    # check for pawn to last row move and prompt player for type of piece conversion wanted
                    return move
            except:
                pass
        return None

    def get_computer_move(self):
        if self.database_path is not None:
            try:

                database = get_database_from_file(self.chessboard.fen(), self.database_path)
                moves, probabilities = get_fen_moves_and_probabilities(database, self.chessboard.fen())
                index = np.searchsorted(probabilities.cumsum(), np.random.rand(), side='left')
                return chess.Move.from_uci(moves[index])
            except:
                pass
        return alpha_beta_move(self.chessboard)

    def onclick(self, event):

        if event.button() == QtCore.Qt.LeftButton:
            if self.chessboard.turn:
                if args.whuman:
                    move =self.human_on_click(event)
                    if move is None:
                        return
                else:
                    move = self.get_computer_move()
                    self.human_first_click = True
                print('white:', str(move))
            else:
                if args.bhuman:
                    move =self.human_on_click(event)
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
        SQUARE_START=40
        SQUARE_SIZE = 125
        SQUARES_PER_ROW_COLUMN = 8
        def get_square_index(pos):
            v=(pos-SQUARE_START)//SQUARE_SIZE
            if 0<=v<SQUARES_PER_ROW_COLUMN:
                return v
            return None

        row = get_square_index(event.x())
        if row is None:
            return None
        row = chr(ord('a')+row)
        col = get_square_index(event.y())
        if col is None:
            return None
        col=SQUARES_PER_ROW_COLUMN-col
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
    counter =0
    def __init__(self, board, alpha, beta, move=None):
        Node.counter +=1
        if Node.counter%1000==0:
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
    counter =0
    def __init__(self, board, parent_node=None, move=None):
        MCTS_Node.counter +=1
        self.won=0
        self.played=0
        self.parent_node = parent_node
        self.move = move
        self.board = board.copy()
        self.legal_moves= [m for m in self.board.legal_moves]
        self.explored_moves = set()
        self.child_ams_heapq = []
        if move is not None:
            self.board.push(self.move)
        self.child_nodes = []

    @staticmethod
    def reset_counter():
        MCTS_Node.counter =0

    def calc_AMS(self):
        if self.parent_node is None:
            return 1
        exploitation = float(self.won)/self.played
        exploration = np.log(self.parent_node.played)/self.played
        return exploitation + np.sqrt(2*exploration)

    def add_new_child(self, move, move_index):
        assert self.legal_moves[move_index]==move
        node = MCTS_Node(self.board, self, move)
        self.child_nodes.append(node)
        self.explored_moves.add(move_index)
        return node

    def get_child_node(self):
        '''
        :return: a child node according to adaptive multi stage sampling
        '''
        NEW_NODE_CHANCE = 0.1
        new_move = False
        if self.child_nodes:
            if np.random.rand()<=NEW_NODE_CHANCE:
                new_move = True
        else:
            new_move = True
        if new_move:
            unexplored_moves =set(np.arange(len(self.legal_moves))).difference(self.explored_moves)
            move_index = unexplored_moves[np.random.randint(len(unexplored_moves))]
            move = self.legal_moves[move_index]
            child_node = self.add_new_child(move, move_index)
        else:
            child_node = heapq.heappop(self.child_ams_heapq)
        return child_node




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

    material_score = (wp-bp) + (wr-br)*5 + (wb-bb)*4 + (wk-bk)*3 + (wq-bq)*9
    score= material_score
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
    if depth>=4:
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
                v = max(v, capturing_moves(child_node.board, child_node, depth+1))
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
                v = min(v, capturing_moves(child_node.board, child_node, depth+1))
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





def mcts_move(board):
    MCTS_Node.reset_counter()
    root = MCTS_Node(board)




def play_random_game(root_node, depth):
    assert depth%2==0, "game must end with opponent turn"

    # todo: add stop condition on win

    current_depth = 0
    node = root_node
    game_nodes = [root_node]
    while current_depth<depth and not node.board.is_checkmate():
        node = node.get_child_node()
        current_depth+=1










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
    dot.node_attr.update(style='filled', fontsize='15', fixedsize='false',fontcolor='blue')
    edges = []
    node_counter = [0]


    def node_to_graph_node(node, dot):
        dot.node(str(node.counter), 'alpha:{0} beta:{1}, move:{2}'.format(node.alpha, node.beta, str(node.move)),
                 shape='box' if node.board.turn else 'oval', color='black' if node.board.turn else 'white')

    def helper(node, node_counter, edges, dot, depth):
        if depth<=0:
            return
        for n in node.child_nodes:
            n.counter =node_counter[0]
            node_counter[0] += 1
            node_to_graph_node(n, dot)
            edges.append((str(node.counter),str(n.counter)))
            helper(n, node_counter, edges, dot, depth-1)

    node.counter = node_counter[0]
    node_to_graph_node(node, dot)
    node_counter[0]+=1
    helper(node, node_counter, edges, dot, depth)
    dot.edges(edges)
    print(dot.source)
    dot.render('test-output/round-table.gv', view=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bhuman', action='store_true')
    parser.add_argument('--whuman', action='store_true')
    parser.add_argument('-memory', help='directory where database files are (dstat[0-9].pkl')
    args = parser.parse_args()
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()