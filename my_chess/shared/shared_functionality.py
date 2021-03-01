from collections import namedtuple
import chess
import numpy
import numpy as np
import os
import pickle
from enum import Enum

# Constants
ROW_SIZE = 8
StatValues = namedtuple("StatValues", ["Wins", "Draws", "Losses"])
TOTAL_QUEEN_MOVES = 56
TOTAL_KNIGHT_MOVES = 8
KNIGHT_MOVES = {(1, 2): 0, (1, -2): 1, (2, 1): 2, (2, -1): 3, (-2, 1): 4, (-2, -1): 5, (-1, -2): 6, (-1, 2): 7}
PLANE_INDEX_TO_KNIGHT_MOVES = dict([(v, k) for k, v in KNIGHT_MOVES.items()])
UNDER_PROMOTIONS = ['r', 'n', 'b']
OUTPUT_PLANES = 73
INPUT_PLANES = 19
PLANE_SYMBOLS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'N', 'W']


class PlaneTypes(Enum):
    NORTH = 0
    NORTH_EAST = 1
    EAST = 2
    SOUTH_EAST = 3
    SOUTH = 4
    SOUTH_WEST = 5
    WEST = 6
    NORTH_WEST = 7


# functions

def position_to_index_1d(pos):
    '''
    :param pos: a string of 2 letters [a-h][1-8]
    :return: the index of the position in the board 0-63
    '''
    res = chess.SQUARE_NAMES.index(pos)
    assert 0 <= res < 64
    return res


def position_to_indices_2d(pos):
    '''
    :param pos: a string of 2 letters [a-h][1-8]
    :return: the 2d indices of the posiition[0-7][0-7]
    '''
    res = (eval(pos[1]) - 1), ord(pos[0]) - ord('a')
    assert 0 <= res[0] <= 7
    assert 0 <= res[1] <= 7
    return res


def index_1d_to_position(index_1d):
    res = chess.SQUARE_NAMES[index_1d]
    return res


def index_1d_to_indices_2d(pos):
    return pos // ROW_SIZE, pos % ROW_SIZE


def indices_2d_to_position(indices_2d):
    index_1d = indices_2d_to_index_1d(indices_2d)
    res = index_1d_to_position(index_1d)
    return res


def indices_2d_to_index_1d(indices_2d):
    return indices_2d[0] * ROW_SIZE + indices_2d[1]


def position_to_mirror_position(pos):
    col = pos[0]
    row = ROW_SIZE - eval(pos[1]) + 1  # mirror row
    promotion = pos[2:]
    new_position = col + str(row) + promotion
    return new_position


def move_to_mirror_move(m):
    return position_to_mirror_position(m[:2]) + position_to_mirror_position(m[2:])


def board_fen_to_hash(fen):
    import hashlib
    m = hashlib.sha256()
    m.update(fen.encode())
    return int.from_bytes(m.digest(), byteorder='little')


def board_fen_to_hash384(fen):
    import hashlib
    m = hashlib.sha384()
    m.update(fen.encode())
    return int.from_bytes(m.digest(), byteorder='little')


def get_fen_moves_and_probabilities(database, baord_fen):
    value = database.get(baord_fen)
    if len(value.keys()) > 0:
        moves_and_probabilities = [(k, v['r']) for k, v in value.items()]
        value = np.array([m[0] for m in moves_and_probabilities])
        probabilities = np.array([m[1] for m in moves_and_probabilities])
        probabilities = np.square(probabilities)  # gives higher probabilities more preference
        probabilities /= probabilities.sum()  # normalize
        return value, probabilities
    return None, None


def get_database_from_file(board_fen, database_path, file_name):
    index1 = board_fen_to_hash(board_fen) % 10
    index2 = board_fen_to_hash384(board_fen) % 10
    with open(os.path.join(database_path, file_name + '{0}_{1}.pkl').format(index1, index2), 'rb') as f:
        database = pickle.load(f)
    return database


def get_nn_moves_and_probabilities(board_list, model, k_best_moves=5):
    from predict import get_input_representation, output_representation_to_moves_and_probabilities, \
        sort_moves_and_probabilities
    input_representation = np.zeros([len(board_list), 8, 8, INPUT_PLANES])
    for i, board in enumerate(board_list):
        board_turn = board.turn
        if not board_turn:
            board = board.mirror()
        input_representation[i] = get_input_representation(board, 0)[np.newaxis]
    output = model.predict(input_representation)
    moves = []
    probabilities = []
    for i in range(output.shape[0]):

        o = output[i]
        sorted_o = np.sort(o.flatten())
        threshold = sorted_o[-k_best_moves]
        a, b, c = np.where(o >= threshold)
        if a.size > k_best_moves:
            high = k_best_moves
            low = 0
            while high-low>1 or a.size> k_best_moves:
                k_best_moves = (low + high) // 2
                threshold = sorted_o[-k_best_moves]
                a, b, c = np.where(o >= threshold)
                if a.size > k_best_moves:
                    high = k_best_moves - 1
                else:
                    low = k_best_moves

        o2 = np.zeros([8, 8, OUTPUT_PLANES])
        o2[a, b, c] = o[a, b, c]
        m, p = output_representation_to_moves_and_probabilities(o2)
        if m.size == 0:
            moves.append([])
            continue
        m, p = sort_moves_and_probabilities(m, p)
        if not board_list[i].turn:
            m = [move_to_mirror_move(move) for move in m]
        moves.append(m)
        probabilities.append(p)
    return moves, probabilities
