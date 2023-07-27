import os
import chess.svg
import time
import argparse
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import QtCore
import json
from datetime import datetime

from algorithms.alpha_beta import alpha_beta_move
from algorithms.mcts import MCTS_Node, mcts_move
from shared.shared_functionality import *
from shared.shared_functionality import get_nn_moves_and_probabilities


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setGeometry(500, 50, 1000, 1000)

        self.widgetSvg = QSvgWidget(parent=self)
        self.widgetSvg.setGeometry(0, 0, 1000, 1000)

        if args.board is not None:
            self.chessboard = chess.Board(args.board)
        else:
            self.chessboard = chess.Board()

        self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg)
        self.widgetSvg.mousePressEvent = self.onclick
        self.human_first_click = True
        self.human_move = ''
        self.nn_model = None
        self.device = None
        self.Log = SingletonLogger().get_logger('play')
        self.Log.setLevel(logging.WARNING)
        if args.debug:
            self.Log.setLevel(logging.DEBUG)
        with open(get_config_path(), 'r') as f:
            self.config = json.load(f)
        self._is_torch_nn = self.config['play']['network_type'] == 'torch'
        self.use_nn = args.nn
        self.use_database = args.database
        self.use_mcts = args.mcts

        self.save_game_path = os.path.join(os.getcwd(), 'game_' + datetime.now().strftime("%d_%m_%Y___%H_%M_%S"))
        self.board_move_counter = 0
        os.makedirs(self.save_game_path)

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
                        print('enter promotion: q, r, b, n')
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

                # change fullmove_number, halfmove_clock temporarily to 0 since these are the fen values in the database.
                fullmove_number = self.chessboard.fullmove_number
                halfmove_clock = self.chessboard.halfmove_clock
                self.chessboard.fullmove_number = 0
                self.chessboard.halfmove_clock = 0
                moves, probabilities = get_fen_moves_and_probabilities(database, self.chessboard.fen())
                self.chessboard.fullmove_number = fullmove_number
                self.chessboard.halfmove_clock = halfmove_clock

                index = np.searchsorted(probabilities.cumsum(), np.random.rand(), side='left')
                return chess.Move.from_uci(moves[index])
            except:
                self.Log
        if self.use_mcts:
            MCTS_Node.use_nn = self.use_nn
            move = mcts_move(self.chessboard, self._is_torch_nn, self.config)[0]
            return move
        if self.use_nn:
            try:
                if self.nn_model is None:
                    if self._is_torch_nn:
                        self.device, self.nn_model = load_pytorch_model(self.config)
                    elif self.config['play']['network_type'] == 'tensorflow':
                        from tensorflow import keras
                        self.nn_model = keras.models.load_model(self.config['play']['tf_nn_path'])
                # returns the best k moves
                moves, probabilities = get_nn_moves_and_probabilities([self.chessboard.copy()],
                                                                      self.nn_model,
                                                                      is_torch_nn=self._is_torch_nn, device=self.device)
                moves = moves[0]  # used batch size 1 so there is only a single array of returned moves.
                probabilities = probabilities[0]
                self.Log.debug(f'moves      : {list(moves)}')
                self.Log.debug(f'probability: {list(probabilities)}')
                for m in moves:
                    if chess.Move.from_uci(m) in self.chessboard.legal_moves:
                        return chess.Move.from_uci(m)
            except:
                self.Log.warning('error while predicting move, skipping...')
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

            self.move(move)

            # save board image to file
            imageVar2 = self.widgetSvg.grab(self.widgetSvg.rect())
            imageVar2.save(os.path.join(self.save_game_path, f'move_{self.board_move_counter}.png'))

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
            self.undo_last_move()
        if event.button() == QtCore.Qt.MiddleButton:
            print(self.chessboard.__repr__())

    def undo_last_move(self):
        self.chessboard.pop()
        self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg)
        # self.widgetSvg.update()
        # time.sleep(1)

    def move(self, move: chess.Move):
        self.chessboard.push(move)
        self.board_move_counter += 1
        self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg)

    def get_human_move(self, event):
        SQUARE_START = 40
        SQUARE_SIZE = 115
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
            print('used alpha beta move')
            self.chessboardSvg = chess.svg.board(self.chessboard).encode("UTF-8")
            self.widgetSvg.load(self.chessboardSvg)
            self.widgetSvg.update()
            time.sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bhuman', action='store_true')
    parser.add_argument('--whuman', action='store_true')
    parser.add_argument('--database', action='store_true', help='get moves from database if available')
    parser.add_argument('--debug', action='store_true', help='sets logging level to debug')
    parser.add_argument('--nn', action='store_true', help='get moves from neural network predictions')
    parser.add_argument('--mcts', action='store_true', help='get moves from monte carlo tree search')
    parser.add_argument('-board', help='start from predefined board (fen), e.g: "8/4Q3/8/7k/5K2/8/8/8 w - - 0 1"')

    args = parser.parse_args()
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
