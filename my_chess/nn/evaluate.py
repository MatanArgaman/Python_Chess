import pickle
import os
import numpy as np
import time
import chess
import json
from tensorflow import keras
import tensorflow as tf
from shared.shared_functionality import get_fen_moves_and_probabilities, get_nn_moves_and_probabilities
from matplotlib.pyplot import *


def evaluate_nn(total_samples=None):
    # 1_6 is test
    with open(os.path.join(os.getcwd(), '../', 'config.json'), 'r') as f:
        config = json.load(f)

    nn_model = keras.models.load_model(config['train']['nn_model_path'])

    counter = [0]
    score_list = []
    for i in range(10):
        for j in range(10):
            score = single_file_evaluate(nn_model, score_list, config, total_samples, counter, i, i, k_best_moves=1)
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


def single_file_evaluate(nn_model, config, total_samples, file_index1, k_best_moves=np.arange(1 , 2)):
    con_data = config['database']
    with open(os.path.join(con_data['file_path'],
                           con_data['file_name'] + '{0}.pkl'.format(file_index1)), 'rb') as f:
        d = pickle.load(f)

    expert_moves_list = []
    board_fen_list = []
    order = np.argsort(k_best_moves)
    k_best_moves = k_best_moves[order]
    indices = np.random.choice(len(d.items()), total_samples, replace=False)
    score_list = [[] for _ in range(k_best_moves.size)]
    for l, (board_fen, v) in enumerate(d.items()):
        if l in indices:
            expert_moves, probabilities = get_fen_moves_and_probabilities(d, board_fen)
            expert_moves = [m[:-1] if 'q' in m else m for m in expert_moves]
            expert_moves_list.append(expert_moves)
            board_fen_list.append(board_fen)

    nn_moves, _ = get_nn_moves_and_probabilities([chess.Board(bf) for bf in board_fen_list], nn_model,
                                                 k_best_moves=k_best_moves[-1])
    for i in range(len(nn_moves)):
        for j, k in enumerate(k_best_moves):
            if set(nn_moves[i][:k]).intersection(expert_moves_list[i]):
                score_list[j].append(1)
            else:
                score_list[j].append(0)
    score_list = np.array([np.mean(item) for item in score_list])
    score_list = score_list[order]
    return score_list


if __name__ == '__main__':
    # print('gpu available:', tf.test.is_gpu_available())
    with open(os.path.join(os.getcwd(), '../', 'config.json'), 'r') as f:
        config = json.load(f)
    # evaluate_nn(total_samples=10002)
    test_index1, test_index2 = config['train']['test_index1'], config['train']['test_index2']

    nn_model = keras.models.load_model(config['train']['nn_model_path'])

    UP_TO_K = 50
    train_score = single_file_evaluate(nn_model, config, 1000, test_index1, k_best_moves=np.arange(1, UP_TO_K))
    test_score = single_file_evaluate(nn_model, config, 1000, test_index1, k_best_moves=np.arange(1, UP_TO_K))
    figure()
    plot(np.arange(1, UP_TO_K), train_score, 'r')
    plot(np.arange(1, UP_TO_K), test_score, 'b')
    show()
