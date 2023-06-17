import argparse
import multiprocessing
import pickle
import tqdm
from multiprocessing import Pool
import chess
import numpy as np
import os
from scipy.sparse import save_npz, csr_matrix

from predict import get_input_representation, get_output_representation
from shared.shared_functionality import board_fen_to_hash, board_fen_to_hash384, position_to_mirror_position, move_to_mirror_move
from pre_processing.A_Data_Statistics.numer_of_games_in_file import number_of_games

def main(in_path, out_path, number_of_files, cpu_count):
    data = []
    for i in range(number_of_files):
        data.append([in_path, out_path, number_of_files, i])
    with Pool(cpu_count) as p:
        for _ in tqdm.tqdm(p.imap(create_win_loss_stats, data), total=len(data)):
            pass


def add_to_stat(stats, result, game):
    board = chess.Board()
    draws = 0
    white_wins = 0
    white_loss = 0
    if result == '1-0':
        white_wins = 1
    elif result == '0-1':
        white_loss = 1
    elif result == '1/2-1/2':
        draws = 1
    for m in game.mainline_moves():
        if m not in board.legal_moves:
            return False
        board.fullmove_number = 0 # todo: make sure to also call this when searching db for moves / nn inference
        board.halfmove_clock = 0
        move_dict = stats.get(board.fen(), {})
        stats[board.fen()] = move_dict
        stat_values = move_dict.get(str(m), {'wins': 0, 'draws': 0, 'losses': 0})
        if board.turn:
            stat_values['wins'] += white_wins
            stat_values['losses'] += white_loss
        else:
            stat_values['wins'] += white_loss
            stat_values['losses'] += white_wins
        stat_values['draws'] += draws
        move_dict[str(m)] = stat_values
        try:
            board.push(m)
        except:
            return False
    return True
    # after final move there are no more moves so there's no reason to record that position


def create_win_loss_stats(indices):
    in_path, out_path, number_of_files, index1 = indices
    failed = total_games = 0
    stats = {}

    IN_FILENAME = "caissabase_{}.pgn"
    OUT_FILENAME = "stat{0}_{1}.pkl"

    in_path = os.path.join(in_path, IN_FILENAME.format(index1))
    total_games = number_of_games(in_path)
    with open(in_path) as pgn:
        with tqdm.tqdm(total=total_games) as pbar:
            while True:
                pbar.update(1)
                try:
                    game = chess.pgn.read_game(pgn)
                except:
                    failed += 1
                    continue
                if game is None:
                    break
                try:
                    result = game.headers['Result']
                except:
                    failed += 1
                    continue
                finished_without_error = add_to_stat(stats, result, game)
                if not finished_without_error:
                    failed += 1

    # split into files by board hash + modulo
    stat_dicts = [{} for _ in range(number_of_files)]
    for fen, move_dict in stats.items():
        board = chess.Board(fen)
        if not board.turn: # learning, and prediction only occurs on white side, we mirror black to white on both cases
            board = board.mirror()
            board.halfmove_clock =0
            board.fullmove_number=0
            fen = board.fen()
            mirror_move_dict = {}
            for m, s in move_dict.items():
                mirror_move_dict[move_to_mirror_move(m)] = s
            move_dict = mirror_move_dict
        hash = board_fen_to_hash(fen)  # make sure fen does not include half moves, full moves or these won't necessarily get the same hash
        index2 = hash % number_of_files
        stat_dicts[index2][fen] = move_dict

    for index2 in range(number_of_files):
        with open(os.path.join(out_path, OUT_FILENAME.format(index1, index2)), 'wb') as fp:
            pickle.dump(stat_dicts[index2], fp)
    print('failed:{0}, processed: {1}, total: {2}'.format(failed,pbar.n, total_games))
    # result int index1 - task index, index2 - hash index
    # a join task should be done on all files with the same index2 value.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path')
    parser.add_argument('out_path')
    parser.add_argument('-number_of_files', help='number of files to split the pgn into',
                        default=multiprocessing.cpu_count() - 1, type=int)
    parser.add_argument('-cpu-count', help='number of cpus to use. Decrease if RAM blows up.',
                        default=multiprocessing.cpu_count() - 1, type=int)

    args = parser.parse_args()
    main(args.in_path, args.out_path, args.number_of_files, args.cpu_count)
