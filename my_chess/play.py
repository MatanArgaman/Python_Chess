import chess.svg
import argparse
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import QtCore
from datetime import datetime
from PyQt5.QtCore import QTimer, QObject, pyqtSignal

from algorithms.mcts import mcts_move, get_nn_and_device
from shared.shared_functionality import *


class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, window):
        super().__init__()
        self.window = window

    def run(self):
        """Long-running task."""
        self.window.play()


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
        self.start_board = self.chessboard.copy()
        self.chessboardSvg = None
        self.update_graphics()
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
        self.use_database = args.database

        self.save_game_path = os.path.join(os.getcwd(), "games",
                                           'game_' + datetime.now().strftime("%d_%m_%Y___%H_%M_%S"))
        self.board_move_counter = 0
        os.makedirs(self.save_game_path)

        self.timer = QTimer()
        self.timer.timeout.connect(self.play)
        self.timer.start(1000)


    def is_flipped_graphics(self):
        if args.whuman and not args.bhuman:
            return args.flip_board
        if args.bhuman and not args.whuman:
            return not args.flip_board
        return False

    def update_graphics(self):
        chessboard_graphics = self.chessboard
        if self.is_flipped_graphics():
            chessboard_graphics = self.chessboard.transform(chess.flip_vertical)
            chessboard_graphics = chessboard_graphics.transform(chess.flip_horizontal)

        last_move = None
        try:
            last_move = self.chessboard.copy().pop()
            if self.is_flipped_graphics():
                last_move = chess.Move.from_uci(move_to_mirror_move(str(last_move), flip_horizontally=True))
        except:
            pass
        self.chessboardSvg = chess.svg.board(chessboard_graphics, lastmove=last_move).encode("UTF-8")
        self.widgetSvg.load(self.chessboardSvg)

    def human_on_click(self, event):

        square = self.human_click_to_square(event)
        if square is None:
            return None
        if self.human_first_click:
            self.human_first_click = False
            self.human_move = square
        else:
            self.human_first_click = True
            self.human_move += square
            try:
                move = None
                if self.is_flipped_graphics():
                    self.human_move = move_to_mirror_move(self.human_move, flip_horizontally=True)

                start_position = position_to_index_1d(self.human_move[:2])
                if self.chessboard.piece_at(start_position) in [chess.Piece.from_symbol('p'),
                                                                chess.Piece.from_symbol('P')]:
                    end_position_row, _ = position_to_indices_2d(self.human_move[2:])
                    if end_position_row in [0, ROW_SIZE - 1]:
                        print('enter promotion: q, r, b, n')
                        value = input()
                        if value in ['q', 'r', 'b', 'n']:
                            move = chess.Move.from_uci(self.human_move + value)
                if move is None:
                    move = chess.Move.from_uci(self.human_move)
                if move in self.chessboard.legal_moves:
                    # check for pawn to last row move and prompt player for type of piece conversion wanted
                    return move
                self.human_first_click = False
                self.human_move = square
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
            except Exception as e:
                self.Log.error("error when using database", str(e))

        if self.nn_model is None:
            self.device, self.nn_model = get_nn_and_device(self.config)
        move = mcts_move(self.chessboard, self.nn_model, self.device)[0]
        return move


    def print_move(self, move, player='white'):
        if self.is_flipped_graphics():
            move = move_to_mirror_move(move, flip_horizontally=True)
        print(f'{player}:', str(move))

    def onclick(self, event):

        if event.button() == QtCore.Qt.LeftButton:
            if self.chessboard.turn:
                if args.whuman:
                    move = self.human_on_click(event)
                    if move is None:
                        return
                else:
                    return
                self.print_move(str(move), player='white')
            else:
                if args.bhuman:
                    move = self.human_on_click(event)
                    if move is None:
                        return
                else:
                    return
                self.print_move(str(move), player='black')

            self.move(move)

            # save board image to file
            self.save_state()

            self.print_if_game_over()

        if event.button() == QtCore.Qt.RightButton:  # undo last move
            self.undo_last_move()
            if (not args.whuman) or (not args.bhuman):
                self.undo_last_move()  # undo twice if there is one or less human players.

        if event.button() == QtCore.Qt.MiddleButton:
            print(self.chessboard.__repr__())

    def print_if_game_over(self):
        if self.chessboard.is_checkmate():
            if self.chessboard.turn:
                print('Black Wins')
            else:
                print('White Wins')
        if self.chessboard.is_insufficient_material():
            print('Draw - insufficient material')
        if self.chessboard.is_stalemate():
            print('Draw - stalemate')
        if self.chessboard.is_repetition():
            print('Draw - by 3 repetition rule')

    def undo_last_move(self):
        if self.chessboard != self.start_board:
            self.chessboard.pop()
            self.update_graphics()

    def move(self, move: chess.Move):
        self.chessboard.push(move)
        self.board_move_counter += 1
        self.update_graphics()

    def human_click_to_square(self, event):
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
        move = None
        if self.chessboard.turn:
            if not args.whuman:
                if not self.chessboard.is_game_over():
                    move = self.get_computer_move()
        else:
            if not args.bhuman:
                if not self.chessboard.is_game_over():
                    move = self.get_computer_move()
        if move is not None:
            self.print_move(str(move), player='white' if self.chessboard.turn else 'black')
            self.move(move)
            self.widgetSvg.update()
        # save board image to file
        self.save_state()
        self.print_if_game_over()

    def save_state(self):
        # save board as image
        imageVar2 = self.widgetSvg.grab(self.widgetSvg.rect())
        imageVar2.save(os.path.join(self.save_game_path, f'move_{self.board_move_counter}.png'))
        # save board as string
        with open(os.path.join(self.save_game_path, f'move_{self.board_move_counter}_fen.txt'), 'w') as fp:
            fp.write(str(self.chessboard.fen()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bhuman', action='store_true', help='black human player')
    parser.add_argument('--whuman', action='store_true', help='white human player')
    parser.add_argument('--flip-board', action='store_true', help='flip the board')
    parser.add_argument('--database', action='store_true', help='get moves from database if available')
    parser.add_argument('--debug', action='store_true', help='sets logging level to debug')
    parser.add_argument('-board', help='start from predefined board (fen), e.g: "8/4Q3/8/7k/5K2/8/8/8 w - - 0 1"')

    args = parser.parse_args()
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
