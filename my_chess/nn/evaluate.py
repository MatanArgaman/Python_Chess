import pickle
import os
import numpy as np
import time
import chess
import json
from tensorflow import keras

from play import get_nn_moves
from shared.shared_functionality import get_fen_moves_and_probabilities


def evaluate_nn():
    # 1_6 is test
    with open(os.path.join(os.getcwd(), '../', 'config.json'), 'r') as f:
        config = json.load(f)

    nn_model = keras.models.load_model(config['train']['nn_model_path'])

    c = 0
    score_list = []
    for i in range(10):
        for j in range(10):
            print('starting file', i, j, time.time())
            con_data = config['database']
            with open(os.path.join(con_data['file_path'],
                                   con_data['file_name'] + '{0}_{1}.pkl'.format(i, j)), 'rb') as f:
                d = pickle.load(f)
                for board_fen, v in d.items():
                    c += 1
                    if c % 100 == 0:
                        score = np.array(score_list)
                        print('score:', score.mean(), score.std())

                    expert_moves, probabilities = get_fen_moves_and_probabilities(d, board_fen)
                    nn_moves = get_nn_moves(chess.Board(board_fen), nn_model, k_best_moves=1)
                    if nn_moves and nn_moves[0] in expert_moves:
                        score_list.append(1)
                    else:
                        score_list.append(0)

                    # expert_set=set(expert_moves)
                    # nn_set=set(nn_moves)
                    # denominator=expert_set.union(nn_set)
                    # nominator=expert_set.intersection(nn_set)
                    # score = len(nominator)/len(denominator)
                    # score_list.append(score)


if __name__ == '__main__':
    evaluate_nn()
