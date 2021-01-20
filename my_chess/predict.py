import chess.pgn
import numpy as np

from shared.shared_functionality import *
import shared

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


def output_representation(moves, probabilities):
    '''

    :param board:
    :param moves:
    :param probabilities: sum to 1
    :return:
    '''

    # parse move
    def parse_single_move(move):
        start_pos_1d = position_to_index(str(move)[:2])
        end_pos_1d = position_to_index(str(move)[2:4])
        start_pos_2d = np.array(position_to_2d_indices(str(move)[:2]))
        end_pos_2d = np.array(position_to_2d_indices(str(move)[2:4]))
        promotion = None
        if len(str(move)) > 4:
            promotion = str(move)[4:].lower()
        return start_pos_1d, end_pos_1d, start_pos_2d, end_pos_2d, promotion

    o = np.zeros([8, 8, 73])

    for i, m in enumerate(moves):
        start_pos_1d, end_pos_1d, start_pos_2d, end_pos_2d, promotion = parse_single_move(m)
        p = probabilities[i]

        dy = end_pos_2d[0] - start_pos_2d[0]
        dx = end_pos_2d[1] - start_pos_2d[1]

        # the first 56 planes encode possible 'queen moves' for any piece:
        # a number of squares [1..7] in which the piece will be moved,
        # along one of eight relative compass directions {N, NE, E, SE, S, SW, W, N W }

        if promotion is None:
            piece = str(b.piece_at(start_pos_1d)).lower()
            if piece != chess.PIECE_SYMBOLS[chess.KNIGHT]:
                # queen moves - this include pawn moves which end in promotion to queen
                if dx != 0 and dy != 0:
                    assert np.abs(dx) == np.abs(dy)
                    steps = np.abs(dx)
                    if dx > 0 and dy > 0:
                        direction = shared.NORTH_EAST
                    elif dx < 0 and dy > 0:
                        direction = shared.NORTH_WEST
                    elif dx > 0 and dy < 0:
                        direction = shared.SOUTH_EAST
                    else:
                        assert dx < 0 and dy < 0
                        direction = shared.SOUTH_WEST
                elif dx != 0:
                    if dx > 0:
                        direction = shared.WEST
                    else:
                        direction = shared.EAST
                    steps = np.abs(dx)
                else:
                    assert dy != 0
                    if dy > 0:
                        direction = shared.NORTH
                    else:
                        direction = shared.SOUTH
                    steps = np.abs(dy)
                plane_index = (steps - 1) * len(shared.PLANE_TYPES) + direction
            else:
                # knight moves
                plane_index = TOTAL_QUEEN_MOVES + KNIGHT_MOVES[(dy, dx)]
        else:
            # under promotions: 3 options (eat diagonally east, eat diagonally west, move up) * promotion options (
            # rock, knight, bishop)
            pawn_move_options = dx + 1  # moves dx from range -1,2 to range 0,3
            plane_index = TOTAL_QUEEN_MOVES + TOTAL_KNIGHT_MOVES + UNDER_PROMOTIONS.index(promotion) * len(
                UNDER_PROMOTIONS) + pawn_move_options
        o[start_pos_2d[0], start_pos_2d[1], plane_index] = p
    return o


if __name__ == '__main__':
    b = chess.Board()
    d = input_representation(b)
