import chess.pgn
import numpy as np

from my_chess.shared import *

# get a set of player games
with open("/home/blacknight/Downloads/Kasparov.pgn") as pgn:
    first_game = chess.pgn.read_game(pgn)
    second_game = chess.pgn.read_game(pgn)
# get board:
board = first_game.board()

# get moves:
moves_generator = first_game.mainline()

# get game meta data:
print(first_game.headers)
# get game winner
print(first_game.headers["Result"])


def input_representation(board, p1_repetitions):
    '''
    The current player is denoted by P1 and the opponent by P2
    :param board:
    :return:
    '''

    def positions_1d_to_2d(pos):
        return pos // ROW_SIZE, pos % ROW_SIZE

    def binary_mask(arr, board, piece, color, mask_index, value):
        l = np.array(list(board.pieces(piece, color)))
        l = positions_1d_to_2d(l)
        arr[..., mask_index][l] = value

    flipped_board = False
    if not board.turn:  # the board is oriented to the perspective of the current player, color is also flipped s.t white is always the current player
        board = board.mirror()
        flipped_board = True

    o = np.zeros([8, 8, 19])
    # p1 piece
    piece_list = [chess.PAWN, chess.BISHOP, chess.KNIGHT, chess.ROOK, chess.QUEEN, chess.KING]
    for i, p in enumerate(piece_list):
        binary_mask(o, board, p, chess.WHITE, i, 1)
    # p2 piece
    for i, p in enumerate(piece_list):
        binary_mask(o, board, p, chess.BLACK, i + len(piece_list), 1)
    # repetitions 12,13
    # p1 repetitions from last turn
    o[..., 13] = p1_repetitions
    # p2 repetitions
    o[..., 13] = 1 if board.is_repetition(count=2) else 0
    # Color 14
    o[..., 14] = o[..., np.arange(6)].sum(axis=2)
    if flipped_board:
        o[..., 14] *= -1
    p2_color = o[..., np.arange(6, 14)].sum(axis=2)
    if not flipped_board:
        p2_color *= -1
    o[..., 14] = o[..., 14] + p2_color

    # total move count 15
    o[..., 15] = board.fullmove_number

    # p1 castling 16
    if board.has_queenside_castling_rights(chess.WHITE):
        o[0, 3, 16] = 1
    if board.has_kingside_castling_rights(chess.WHITE):
        o[0, 4, 16] = 1
    # p2 castling 17
    if board.has_queenside_castling_rights(chess.BLACK):
        o[7, 3, 17] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        o[7, 4, 17] = 1
    # No progress count 18
    o[..., 15] = board.halfmove_clock
    return o


def output_representation(board, moves, probabilities):
    # parse move
    def parse_single_move(move):
        start_pos = position_to_2d_indices(move.__str__()[:2])
        end_pos = position_to_2d_indices(move.__str__()[2:4])
        promotion = None
        if len(str(move))>4:
            promotion = str(move)[4:].lower()
        return start_pos, end_pos, promotion

    o = np.zeros([8, 8, 73])




    # the first 56 planes encode possible 'queen moves' for any piece:
    # a number of squares [1..7] in which the piece will be moved,
    # along one of eight relative compass directions {N, N E, E, SE, S, SW, W, N W }




if __name__ == '__main__':
    b = chess.Board()
    d = input_representation(b)
