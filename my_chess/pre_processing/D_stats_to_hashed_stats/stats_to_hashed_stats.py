# regroups stats files into original number of split files using hash + modulo
import argparse
import pickle
from pathlib import Path
import os
import re


def main(in_path, out_path):
    IN_FILENAME = "stat{0}_{1}.pkl"
    OUT_FILENAME = "cstat{0}.pkl"

    stat_files =[f for f in Path(in_path).rglob("stat*.pkl")]
    same_index2_files = {}
    for f in stat_files:
        filename = os.path.basename(f)
        m = re.search("stat([0-9]*)_([0-9]*).pkl",filename)
        # index1 = m.group(1)
        index2 = m.group(2)
        same_index2_files[index2] = same_index2_files.get(index2, [])
        same_index2_files[index2].append(filename)
    for index2, filename_list in same_index2_files.items():
        full_path = os.path.join(in_path, filename_list[0])
        with open(full_path, 'rb') as fp:
            stat_dict = pickle.load(fp)
        for filename in filename_list[1:]:
            full_path = os.path.join(in_path, filename)
            with open(full_path, 'rb') as fp:
                additional_stat_dict = pickle.load(fp)
                for board_fen, move_dict in additional_stat_dict.items():
                    if board_fen not in stat_dict:
                        stat_dict[board_fen]=move_dict
                    else:
                        # join stats
                        stat_fen = stat_dict[board_fen]
                        for move, win_loss_draw in move_dict.items():
                            if move not in stat_fen:
                                stat_fen[move] = win_loss_draw
                            else:
                                stat_fen[move]['wins']+=win_loss_draw['wins']
                                stat_fen[move]['draws']+=win_loss_draw['draws']
                                stat_fen[move]['losses']+=win_loss_draw['losses']
        out_path = os.path.join(out_path, OUT_FILENAME.format(index2))
        with open(out_path, "wb") as fp:
            pickle.dump(stat_dict, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_path')
    parser.add_argument('out_path')
    args = parser.parse_args()
    main(args.in_path,args.out_path)
