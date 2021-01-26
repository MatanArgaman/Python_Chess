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

            expert_moves_list = []
            board_fen_list = []
            for l, (board_fen, v) in enumerate(d.items()):
                expert_moves, probabilities = get_fen_moves_and_probabilities(d, board_fen)
                expert_moves_list.append(expert_moves)
                board_fen_list.append(board_fen)

                c += 1
                if c % 10000 == 0 or l == len(d.keys()) - 1:
                    nn_moves = get_nn_moves([chess.Board(bf) for bf in board_fen_list], nn_model, k_best_moves=1)
                    for k in range(len(nn_moves)):
                        if nn_moves[k] and nn_moves[k][0] in expert_moves_list[k]:
                            score_list.append(1)
                        else:
                            score_list.append(0)

                    score = np.array(score_list)
                    print('score:', score.mean())
                    expert_moves_list = []
                    board_fen_list = []

                # expert_set=set(expert_moves)
                # nn_set=set(nn_moves)
                # denominator=expert_set.union(nn_set)
                # nominator=expert_set.intersection(nn_set)
                # score = len(nominator)/len(denominator)
                # score_list.append(score)


if __name__ == '__main__':
    evaluate_nn()
