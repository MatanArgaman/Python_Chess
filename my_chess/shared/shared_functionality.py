from collections import namedtuple
import chess
import numpy as np
import os
import pickle

# Constants
ROW_SIZE = 8
StatValues = namedtuple("StatValues",["Wins", "Draws", "Losses"])
TOTAL_QUEEN_MOVES = 56
TOTAL_KNIGHT_MOVES =  8
KNIGHT_MOVES = {(1, 2): 0, (1, -2): 1, (2, 1): 2, (2, -1): 3, (-2, 1): 4, (-2, -1): 5, (-1, -2): 6, (-1, 2): 7}
PLANE_INDEX_TO_KNIGHT_MOVES= dict([(v,k) for k,v in KNIGHT_MOVES.items()])
UNDER_PROMOTIONS=['r', 'n', 'b']
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


def indices_2d_to_position(indices_2d):
    index_1d = indices_2d_to_index_1d(indices_2d)
    res = index_1d_to_position(index_1d)
    return res

def indices_2d_to_index_1d(indices_2d):
    return indices_2d[0] * ROW_SIZE + indices_2d[1]


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
    moves = database.get(baord_fen)
    if len(moves.keys()) > 0:
        moves_and_probabilities = [(k, v['r']) for k, v in moves.items()]
        moves = np.array([m[0] for m in moves_and_probabilities])
        probabilities = np.array([m[1] for m in moves_and_probabilities])
        probabilities = np.square(probabilities)  # gives higher probabilities more preference
        probabilities /= probabilities.sum() # normalize
        return moves, probabilities
    return None, None

def get_database_from_file(board_fen, database_path):
    index1 = board_fen_to_hash(board_fen) % 10
    index2 = board_fen_to_hash384(board_fen) % 10
    with open(os.path.join(database_path, 'dstat{0}_{1}.pkl').format(index1, index2), 'rb') as f:
        database = pickle.load(f)
    return database
