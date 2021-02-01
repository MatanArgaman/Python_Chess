import pickle
import os
import numpy as np
import time
import chess
import json
from tensorflow import keras
import tensorflow as tf
from play import get_nn_moves_and_probabilities
from shared.shared_functionality import get_fen_moves_and_probabilities


def evaluate_nn(total_samples = None):
    # 1_6 is test
    with open(os.path.join(os.getcwd(), '../', 'config.json'), 'r') as f:
        config = json.load(f)

    nn_model = keras.models.load_model(config['train']['nn_model_path'])

    counter = [0]
    score_list = []
    for i in range(10):
        for j in range(10):
            score = single_file_evaluate(nn_model, score_list, config, total_samples, counter, i,i, k_best_moves=5)
            print("score:", score)
            counter = [0]
            score_list = []
            # print('starting file', i, j, time.time())
            # expert_set=set(expert_moves)
            # nn_set=set(nn_moves)
            # denominator=expert_set.union(nn_set)
            # nominator=expert_set.intersection(nn_set)
            # score = len(nominator)/len(denominator)
            # score_list.append(score)


def single_file_evaluate(nn_model, score_list, config, total_samples, counter, file_index1, file_index2, k_best_moves=1):
    con_data = config['database']
    with open(os.path.join(con_data['file_path'],
                           con_data['file_name'] + '{0}_{1}.pkl'.format(file_index1, file_index2)), 'rb') as f:
        d = pickle.load(f)

    expert_moves_list = []
    board_fen_list = []
    for l, (board_fen, v) in enumerate(d.items()):
        expert_moves, probabilities = get_fen_moves_and_probabilities(d, board_fen)
        expert_moves= [m[:-1] if 'q' in m else m for m in expert_moves]
        expert_moves_list.append(expert_moves)
        board_fen_list.append(board_fen)

        counter[0] += 1
        if counter[0] % 10000 == 0 or l == len(d.keys()) - 1:
            nn_moves, _ = get_nn_moves_and_probabilities([chess.Board(bf) for bf in board_fen_list], nn_model,
                                                         k_best_moves=k_best_moves)
            for k in range(len(nn_moves)):
                try:
                    if set(nn_moves[k]).intersection(expert_moves_list[k]):
                        score_list.append(1)
                    else:
                        score_list.append(0)
                except:
                    pass

            # score = np.array(score_list)
            # print('score:', score.mean())
            expert_moves_list = []
            board_fen_list = []
        if total_samples is not None and counter[0] > total_samples:
            return np.array(score_list).mean()


if __name__ == '__main__':
    print('gpu available:', tf.test.is_gpu_available())
    evaluate_nn(total_samples=10002)
