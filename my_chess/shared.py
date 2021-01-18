
from collections import namedtuple
import chess

# Constants
ROW_SIZE = 8
StatValues = namedtuple("StatValues",["Wins", "Draws", "Losses"])

# functions

def position_to_index(pos):
    '''
    :param pos: a string of 2 letters [a-h][1-8]
    :return: the index of the position in the board 0-63
    '''
    res = chess.SQUARE_NAMES.index(pos)
    assert 0 <= res < 64
    return res


def position_to_2d_indices(pos):
    '''
    :param pos: a string of 2 letters [a-h][1-8]
    :return: the 2d indices of the posiition[0-7][0-7]
    '''
    res = (eval(pos[1]) - 1), ord(pos[0]) - ord('a')
    assert 0 <= res[0] <= 7
    assert 0 <= res[1] <= 7
    return res

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
