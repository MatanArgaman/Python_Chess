import argparse
import pickle
import numpy as np
import chess
from tqdm import tqdm
from pathlib import Path

from predict import get_output_representation, get_input_representation, output_representation_to_moves_and_probabilities
from shared.shared_functionality import move_to_mirror_move, INPUT_PLANES, OUTPUT_PLANES, get_move_value

def main(dir_path):
    IN_FILENAME = "dstat{0}.pkl"
    for f in Path(dir_path).rglob('dstat*.pkl'):
        print(f'file:{str(f)}')
        with open(str(f), 'rb') as fp:
            d = pickle.load(fp)
        value = np.zeros([len(d.items())], dtype=float)
        skipped_boards = 0
        skipped_boards_no_wins = 0
        skipped_boards_representation_error = 0
        skipped_boards_no_probabilities = 0
        current_index = 0
        for i, (fen, item) in tqdm(enumerate(d.items()), total=len(d.keys())):
            b = chess.Board(fen)
            moves_and_probabilities = []
            wins = 0
            draws = 0
            losses = 0
            for move, v in item.items():
                moves_and_probabilities.append((move, get_move_value(v)))
                wins += v['wins']
                losses += v['losses']
                draws += v['draws']

            played = wins + losses + draws
            if played == 0:
                skipped_boards_no_wins += 1
                skipped_boards += 1
                continue
            probabilities = np.array([m[1] for m in moves_and_probabilities])
            if probabilities.sum() == 0:
                skipped_boards_no_probabilities += 1
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
                #input_representation = get_input_representation(b, 0)
                output_representation = get_output_representation(moves, probabilities, b)
                mp = output_representation_to_moves_and_probabilities(output_representation)
            except:
                skipped_boards_representation_error+=1
                skipped_boards += 1
                continue
            for index1, m in enumerate(moves):
                if len(m)>4:
                    if m[4]=='q': # move to queen promotion is inferred implicitly
                        m = m[:4]
                index2 = list(mp[0]).index(m)
                assert probabilities[index1] == mp[1][index2]

            current_index += 1  # make sure this is the last line in the loop (not continues after it)
        print("skipped total {0}/{1}".format(skipped_boards, len(d.items())))
        print('skipped details:')
        print("skipped no wins  {0}/{1}".format(skipped_boards_no_wins, len(d.items())))
        print("skipped no probabilities  {0}/{1}".format(skipped_boards_no_probabilities, len(d.items())))
        print("skipped representation error {0}/{1}".format(skipped_boards_representation_error, len(d.items())))
        assert skipped_boards_no_probabilities == 0 and skipped_boards_representation_error ==0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_path')
    args = parser.parse_args()
    main(args.dir_path)