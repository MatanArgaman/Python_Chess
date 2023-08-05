import argparse
import multiprocessing
import os
import pickle
import re
from multiprocessing import Pool
import numpy as np
import tqdm
import chess
from scipy.sparse import csr_matrix, save_npz
from pathlib import Path

from predict import get_output_representation, get_input_representation
from shared.shared_functionality import move_to_mirror_move, INPUT_PLANES, OUTPUT_PLANES, get_move_value


def main(path, cpu_count):
    in_files = [str(f) for f in Path(path).rglob("dstat*.pkl")]
    # create_input_output_representation_with_win_probability(in_files[36])
    with Pool(cpu_count) as p:
        for _ in tqdm.tqdm(p.imap(create_input_output_representation_with_win_probability, in_files),
                           total=len(in_files)):
            pass


def create_input_output_representation_with_win_probability(path):
    IN_FILENAME = "dstat{0}.pkl"
    filename = os.path.basename(path)
    m = re.search("dstat([0-9]*).pkl", filename)
    index1 = m.group(1)
    with open(path, 'rb') as f:
        d = pickle.load(f)
    input_arr = np.zeros([len(d.items()), 8, 8, INPUT_PLANES], dtype=float)
    output_arr = np.zeros([len(d.items()), 8, 8, OUTPUT_PLANES], dtype=float)
    value = np.zeros([len(d.items())], dtype=float)
    mask_arr = np.zeros([len(d.items())], dtype=bool)
    skipped_boards = 0
    skipped_boards_no_stats = 0
    skipped_moves_no_stats = 0
    skipped_boards_representation_error = 0
    current_index = 0
    for i, (fen, item) in enumerate(d.items()):
        assert fen.endswith('0'), "remove when fixing this in both dataset and predict"
        board = chess.Board(fen)
        board.fullmove_number = 0
        board.halfmove_clock = 0

        if not board.turn:  # if black's turn then mirror board and moves
            raise Exception(
                'data at this point should all be white turn as black turn state have already been mirrored')

        moves_and_probabilities = []
        wins = 0
        draws = 0
        losses = 0
        has_winning_moves = False
        for move, move_stats in item.items():
            if (move_stats['wins'] + move_stats['losses'] + move_stats['draws']) == 0:
                skipped_moves_no_stats += 1
                continue
            v = get_move_value(move_stats)
            if v > 1e-5:
                has_winning_moves = True
            moves_and_probabilities.append((move, v))
            wins += move_stats['wins']
            losses += move_stats['losses']
            draws += move_stats['draws']
        if not has_winning_moves:
            mask_arr[current_index] = True
        played = wins + losses + draws
        if played == 0:
            skipped_boards_no_stats += 1
            skipped_boards += 1
            continue

        value[current_index] = (float(wins - losses) / played)  # range from -1 to 1

        if has_winning_moves:
            only_winning_moves_and_probabilities = []
            for m in moves_and_probabilities:
                if m[1] > 0:
                    only_winning_moves_and_probabilities.append(m)
            moves_and_probabilities = only_winning_moves_and_probabilities

        probabilities = np.array([m[1] for m in moves_and_probabilities])
        moves = np.array([m[0] for m in moves_and_probabilities])

        if not has_winning_moves:
            probabilities[np.abs(probabilities) < 1e-5] = 1e-2
            probabilities = 1.0 / np.abs(probabilities)
        probabilities = np.square(probabilities)  # gives higher probabilities more preference
        probabilities /= probabilities.sum()  # normalize
        try:
            input_arr[current_index] = get_input_representation(board, 0)
            output_arr[current_index] = get_output_representation(moves, probabilities, board)
        except:
            skipped_boards_representation_error += 1
            skipped_boards += 1
            continue
        current_index += 1  # make sure this is the last line in the loop (not continues after it)
    print("\nskipped total {0}/{1}".format(skipped_boards, len(d.items())))
    print('skipped details:')
    print("skipped no board stats {0}/{1}".format(skipped_boards_no_stats, len(d.items())))
    print("skipped representation error {0}/{1}".format(skipped_boards_representation_error, len(d.items())))
    print("skipped no move stats {0}".format(skipped_moves_no_stats))
    save_results_to_files(path,
                          input_arr[:current_index],
                          output_arr[:current_index],
                          value[:current_index],
                          mask_arr[:current_index],
                          index1)


def save_results_to_files(path, input_arr, output_arr, value, mask_arr, index1):
    OUT_FILENAME_INPUT_REPRESENTATION = "mstat{0}_i.npz"
    OUT_FILENAME_OUTPUT_REPRESENTATION = "mstat{0}_o.npz"
    OUT_FILENAME_OUTPUT_SIZE = "mstat{0}_s.pkl"
    OUT_FILENAME_OUTPUT_VALUES = "mstat{0}_v.pkl"
    OUT_FILENAME_OUTPUT_MASK = "mstat{0}_m.pkl"

    path = os.path.dirname(path)
    sparse_output_arr = csr_matrix(output_arr.reshape([output_arr.shape[0], -1]))
    sparse_input_arr = csr_matrix(input_arr.reshape([input_arr.shape[0], -1]))
    save_npz(os.path.join(path, OUT_FILENAME_INPUT_REPRESENTATION.format(index1)), sparse_input_arr)
    save_npz(os.path.join(path, OUT_FILENAME_OUTPUT_REPRESENTATION.format(index1)), sparse_output_arr)

    samples = input_arr.shape[0]
    assert input_arr.shape[0] == output_arr.shape[0]
    with open(os.path.join(path, OUT_FILENAME_OUTPUT_SIZE.format(index1)), 'wb') as fp:
        pickle.dump(samples, fp)
    with open(os.path.join(path, OUT_FILENAME_OUTPUT_VALUES.format(index1)), 'wb') as fp:
        pickle.dump(value, fp)
    with open(os.path.join(path, OUT_FILENAME_OUTPUT_MASK.format(index1)), 'wb') as fp:
        pickle.dump(mask_arr, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('-cpu-count', help='number of cpus to use. Decrease if RAM blows up.',
                        default=multiprocessing.cpu_count() - 1, type=int)

    args = parser.parse_args()
    main(args.path, args.cpu_count)
