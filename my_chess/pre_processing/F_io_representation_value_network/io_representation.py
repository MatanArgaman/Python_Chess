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
    value = np.zeros([len(d.items())], dtype=float)
    skipped_boards = 0
    skipped_boards_representation_error = 0
    current_index = 0
    for i, (fen, item) in enumerate(d.items()):
        assert fen.endswith('0'), "remove when fixing this in both dataset and predict"
        board = chess.Board(fen)
        board.fullmove_number = 0
        board.halfmove_clock = 0
        wins = 0
        draws = 0
        losses = 0
        for move, v in item.items():
            wins += v['wins']
            losses += v['losses']
            draws += v['draws']

        played = wins + losses + draws
        if played == 0:
            skipped_boards += 1
            continue

        value[current_index] = (float(wins - losses) / played)  # range from -1 to 1
        assert 1 >= value[current_index] >= -1, f"illegal value {value[current_index]}"
        if not board.turn:  # if black's turn then mirror board and moves
            raise Exception(
                'data at this point should all be white turn as black turn state have already been mirrored')
        try:
            input_arr[current_index] = get_input_representation(board, 0)
        except:
            skipped_boards_representation_error += 1
            skipped_boards += 1
            continue
        current_index += 1  # make sure this is the last line in the loop (not continues after it)
    print("skipped total {0}/{1}".format(skipped_boards, len(d.items())))
    print('skipped details:')
    print("skipped representation error {0}/{1}".format(skipped_boards_representation_error, len(d.items())))
    save_results_to_files(path,
                          input_arr[:current_index],
                          value[:current_index],
                          index1)


def save_results_to_files(path, input_arr, value, index1):
    OUT_FILENAME_INPUT_REPRESENTATION = "vstat{0}_i.npz"
    OUT_FILENAME_OUTPUT_VALUE = "vstat{0}_o.npy"
    OUT_FILENAME_OUTPUT_SIZE = "vstat{0}_s.pkl"

    path = os.path.dirname(path)
    sparse_input_arr = csr_matrix(input_arr.reshape([input_arr.shape[0], -1]))
    save_npz(os.path.join(path, OUT_FILENAME_INPUT_REPRESENTATION.format(index1)), sparse_input_arr)

    samples = input_arr.shape[0]
    np.save(os.path.join(path, OUT_FILENAME_OUTPUT_VALUE.format(index1)), value)
    with open(os.path.join(path, OUT_FILENAME_OUTPUT_SIZE.format(index1)), 'wb') as fp:
        pickle.dump(samples, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('-cpu-count', help='number of cpus to use. Decrease if RAM blows up.',
                        default=multiprocessing.cpu_count() - 1, type=int)

    args = parser.parse_args()
    main(args.path, args.cpu_count)
