import chess
import chess.svg
import numpy as np
import re
import time
import argparse

from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import QTimer
from PyQt5 import QtCore


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
                move = chess.Move.from_uci(self.human_move)
                if move in self.chessboard.legal_moves:
                    return move
            except:
                pass
        return None

    def onclick(self, event):

        if event.button() == QtCore.Qt.LeftButton:
            if self.chessboard.turn:
                if args.whuman:
                    move =self.human_on_click(event)
                    if move is None:
                        return
                else:
                    move = alpha_beta_move(self.chessboard)
                    self.human_first_click = True
                print('white:', str(move))
            else:
                if args.bhuman:
                    move =self.human_on_click(event)
                    if move is None:
                        return
                else:
                    move = alpha_beta_move(self.chessboard)
                    self.human_first_click = True
                print('black:', str(move))

            self.chessboard.push(move)
            self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
            self.widgetSvg.load(self.chessboardSvg)
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
        self.alpha = alpha
        self.beta = beta
        self.move = move
        self.board = board.copy()
        if move is not None:
            self.board.push(self.move)
        self.child_nodes = []
    @staticmethod
    def reset_counter():
        Node.counter =0

class MCTS_Node:
    counter =0
    def __init__(self, board, move=None):
        Node.counter +=1
        self.won=0
        self.played=0

    @staticmethod
    def reset_counter():
        Node.counter =0



def basic_evaluation(board):
    if board.is_checkmate():
        if board.turn:
            return np.inf
        else:
            return -np.inf
    d_white = {'P': 1, 'R': 5, 'B': 4, 'N': 3, 'Q': 9}
    d_black = [(k.lower(), -v) for k, v in d_white.items()]
    d = d_white
    d.update(d_black)
    b = re.split('\s|\n', str(board))
    score = sum([d.get(l, 0) for l in b])
    return score


def alpha_beta(board, depth, node):
    if depth == 0:
        v=basic_evaluation(node.board)
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


def mcts_move(board):
    MCTS_Node.reset_counter()




def alpha_beta_move(board):
    Node.reset_counter()
    root = Node(board, -np.inf, np.inf)
    v = alpha_beta(board, 4, root)  # must be an even number to end turn in opponent's turn.
    root.child_nodes.sort(key=lambda x: -x.beta if board.turn else x.alpha)
    print('total nodes explored:', Node.counter)
    # equivalent_moves = [root.child_nodes[0]]
    # for i in range(1, len(root.child_nodes)):
    #     if board.turn:
    #         if root.child_nodes[0].beta == root.child_nodes[i].beta:
    #             equivalent_moves.append(root.child_nodes[i])
    #     else:
    #         if root.child_nodes[0].alpha == root.child_nodes[i].alpha:
    #             equivalent_moves.append(root.child_nodes[i])
    # return equivalent_moves[np.random.randint(0, len(equivalent_moves))].move
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
    args = parser.parse_args()
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
