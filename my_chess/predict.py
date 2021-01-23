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


def get_input_representation(board, p1_repetitions):
    '''
    The current player is denoted by P1 and the opponent by P2
    :param board:
    :return:
    '''


    def binary_mask(arr, board, piece, color, mask_index, value):
        l = np.array(list(board.pieces(piece, color)), dtype=np.int)
        l = index_1d_to_indices_2d(l)
        arr[..., mask_index][l] = value

    flipped_board = False
    if not board.turn:  # the board is oriented to the perspective of the current player, color is also flipped s.t white is always the current player
        board = board.mirror()
        flipped_board = True

    o = np.zeros([8, 8, INPUT_PLANES])
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


def get_output_representation(moves, probabilities, board):
    '''

    :param board:
    :param moves:
    :param probabilities: sum to 1
    :return:
    '''

    # parse move
    def parse_single_move(move):
        start_index_1d = position_to_index_1d(str(move)[:2])
        end_index_1d = position_to_index_1d(str(move)[2:4])
        start_indices_2d = np.array(position_to_indices_2d(str(move)[:2]))
        end_indices_2d = np.array(position_to_indices_2d(str(move)[2:4]))
        promotion = None
        if len(str(move)) > 4:
            promotion = str(move)[4:].lower()
        return start_index_1d, end_index_1d, start_indices_2d, end_indices_2d, promotion

    o = np.zeros([8, 8, OUTPUT_PLANES])

    for i, m in enumerate(moves):
        start_index_1d, end_index_1d, start_indices_2d, end_indices_2d, promotion = parse_single_move(m)
        p = probabilities[i]

        dr = end_indices_2d[0] - start_indices_2d[0]
        dc = end_indices_2d[1] - start_indices_2d[1]

        # the first 56 planes encode possible 'queen moves' for any piece:
        # a number of squares [1..7] in which the piece will be moved,
        # along one of eight relative compass directions {N, NE, E, SE, S, SW, W, N W }

        if (promotion is None) or (promotion==chess.PIECE_SYMBOLS[chess.QUEEN]):
            piece = str(board.piece_at(start_index_1d)).lower()
            if piece != chess.PIECE_SYMBOLS[chess.KNIGHT]:
                # queen moves - this include pawn moves which end in promotion to queen
                if dc != 0 and dr != 0:
                    assert np.abs(dc) == np.abs(dr)
                    steps = np.abs(dc)
                    if dc > 0 and dr > 0:
                        direction = PlaneTypes.NORTH_EAST.value
                    elif dc < 0 and dr > 0:
                        direction = PlaneTypes.NORTH_WEST.value
                    elif dc > 0 and dr < 0:
                        direction = PlaneTypes.SOUTH_EAST.value
                    else:
                        assert dc < 0 and dr < 0
                        direction = PlaneTypes.SOUTH_WEST.value
                elif dc != 0:
                    if dc > 0:
                        direction = PlaneTypes.WEST.value
                    else:
                        direction = PlaneTypes.EAST.value
                    steps = np.abs(dc)
                else:
                    assert dr != 0
                    if dr > 0:
                        direction = PlaneTypes.NORTH.value
                    else:
                        direction = PlaneTypes.SOUTH.value
                    steps = np.abs(dr)
                plane_index = (steps - 1) * len(PlaneTypes) + direction
            else:
                # knight moves
                plane_index = TOTAL_QUEEN_MOVES + KNIGHT_MOVES[(dr, dc)]
        else:
            # under promotions: 3 options (eat diagonally east, eat diagonally west, move up) * promotion options (
            # rock, knight, bishop)
            assert dr==1
            pawn_move_options = dc + 1  # moves dc from range -1,2 to range 0,3
            plane_index = TOTAL_QUEEN_MOVES + TOTAL_KNIGHT_MOVES + UNDER_PROMOTIONS.index(promotion) * len(
                UNDER_PROMOTIONS) + pawn_move_options
        o[start_indices_2d[0], start_indices_2d[1], plane_index] = p
    return o

def sort_moves_and_probabilities(moves,probabilities):
    order = np.argsort(probabilities)[::-1] # descending order
    probabilities=probabilities[order]
    moves = moves[order]
    return moves, probabilities

def output_representation_to_moves_and_probabilities(output_representation):
    '''
    :param output_representation:

    *Note that queen promotion moves do note have 'q' (chess.PIECE_SYMBOLS[chess.QUEEN]) but that should be inferred by
     pawn movement + lack of under promotion

    :return:
    '''
    o = output_representation
    moves = []
    probabilities = []
    for row, column, plane_index in list(zip(*np.where(o))):
        probabilities.append(o[row, column, plane_index])
        start_pos = indices_2d_to_position([row, column])
        assert plane_index >= 0
        promotion=''
        if plane_index < TOTAL_QUEEN_MOVES:
            steps = plane_index // len(PlaneTypes) + 1
            direction = plane_index % len(PlaneTypes)
            if direction == PlaneTypes.NORTH.value:
                dc = 0
                dr = steps
            elif direction == PlaneTypes.NORTH_EAST.value:
                dc = steps
                dr = steps
            elif direction == PlaneTypes.WEST.value:
                dc = steps
                dr = 0
            elif direction == PlaneTypes.SOUTH_WEST.value:
                dc = steps
                dr = -steps
            elif direction == PlaneTypes.SOUTH.value:
                dc = 0
                dr = -steps
            elif direction == PlaneTypes.SOUTH_EAST.value:
                dc = -steps
                dr = -steps
            elif direction == PlaneTypes.EAST.value:
                dc = -steps
                dr = 0
            else:
                assert direction == PlaneTypes.NORTH_WEST.value
                dc = -steps
                dr = steps
        elif plane_index < TOTAL_QUEEN_MOVES + TOTAL_KNIGHT_MOVES:
            dr, dc = PLANE_INDEX_TO_KNIGHT_MOVES[plane_index - TOTAL_QUEEN_MOVES]
        else:
            assert plane_index < o.shape[2]
            plane_index = plane_index - TOTAL_QUEEN_MOVES - TOTAL_KNIGHT_MOVES
            dr = 1
            promotion = UNDER_PROMOTIONS[plane_index//len(UNDER_PROMOTIONS)]
            dc = (plane_index % len(UNDER_PROMOTIONS)) - 1 # moves dc from range 0,3 to range -1,2
        end_pos = indices_2d_to_position([row + dr, column + dc])
        moves.append(start_pos + end_pos + promotion)
    return np.array(moves), np.array(probabilities)

if __name__ == '__main__':
    b = chess.Board()
    d = get_input_representation(b)
