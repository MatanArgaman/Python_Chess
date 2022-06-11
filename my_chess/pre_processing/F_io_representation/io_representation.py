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
        for _ in tqdm.tqdm(p.imap(create_input_output_representation_with_win_probability, in_files), total=len(in_files)):
            pass


def create_input_output_representation_with_win_probability(path):
    IN_FILENAME = "dstat{0}.pkl"
    filename = os.path.basename(path)
    m = re.search("dstat([0-9]*).pkl", filename)
    index1 = m.group(1)
    with open(path, 'rb') as f:
        d = pickle.load(f)
    input_arr = np.zeros([8, 8, INPUT_PLANES * len(d.items())], dtype=float)
    output_arr = np.zeros([8, 8, OUTPUT_PLANES * len(d.items())], dtype=float)
    value = np.zeros([len(d.items())], dtype=float)
    skipped_boards = 0
    current_index = 0
    for i, (fen, item) in enumerate(d.items()):
        b = chess.Board(fen)
        moves_and_probabilities = []
        wins = 0
        draws = 0
        losses = 0
        for move, v in item.items():
            if float(v['wins'] - v['losses']) > 0:
                moves_and_probabilities.append((move, get_move_value(v)))
                wins += v['wins']
                losses += v['losses']
                draws += v['draws']

        played = wins + losses + draws
        if played == 0:
            skipped_boards += 1
            continue
        probabilities = np.array([m[1] for m in moves_and_probabilities])
        if probabilities.sum() == 0:
            skipped_boards += 1
            continue

        value[current_index] = (float(wins - losses) / played)  # range from -1 to 1

        if not b.turn:  # if black's turn then mirror board and moves
            b = b.mirror()
            moves = np.array([move_to_mirror_move(m[0]) for m in moves_and_probabilities])
        else:
            moves = np.array([m[0] for m in moves_and_probabilities])
        probabilities = np.square(probabilities)  # gives higher probabilities more preference
        probabilities /= probabilities.sum()  # normalize
        try:
            input_arr[..., current_index * INPUT_PLANES:(current_index + 1) * INPUT_PLANES] = get_input_representation(b, 0)
            output_arr[..., current_index * OUTPUT_PLANES:(current_index + 1) * OUTPUT_PLANES] = get_output_representation(moves, probabilities, b)
        except:
            skipped_boards += 1
            continue
        current_index += 1 # make sure this is the last line in the loop (not continues after it)
    print("skipped {0}/{1}".format(skipped_boards, len(d.items())))
    save_results_to_files(path,
                          input_arr[...,:current_index * INPUT_PLANES],
                          output_arr[...,:current_index * OUTPUT_PLANES],
                          value[...,:current_index],
                          index1)


def save_results_to_files(path, input_arr, output_arr, value, index1):
    OUT_FILENAME_INPUT_REPRESENTATION = "estat{0}_i.npz"
    OUT_FILENAME_OUTPUT_REPRESENTATION = "estat{0}_o.npz"
    OUT_FILENAME_OUTPUT_VALUES = "estat{0}_v.npz"

    path = os.path.dirname(path)
    sparse_output_arr = csr_matrix(output_arr.reshape([8, -1]))
    sparse_input_arr = csr_matrix(input_arr.reshape([8, -1]))
    save_npz(os.path.join(path, OUT_FILENAME_INPUT_REPRESENTATION.format(index1)), sparse_input_arr)
    save_npz(os.path.join(path, OUT_FILENAME_OUTPUT_REPRESENTATION.format(index1)), sparse_output_arr)
    with open(os.path.join(path, OUT_FILENAME_OUTPUT_VALUES.format(index1)), 'wb') as fp:
        pickle.dump(value, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('-cpu-count', help='number of cpus to use. Decrease if RAM blows up.',
                        default=multiprocessing.cpu_count() - 1, type=int)

    args = parser.parse_args()
    main(args.path, args.cpu_count)
