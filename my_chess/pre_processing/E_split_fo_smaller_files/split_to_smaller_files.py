# regroups stats files into original number of split files using hash + modulo
# if the next part (io_representation) results in no memory left, increase number_of_files value.
import argparse
import pickle
from itertools import islice
from pathlib import Path
import os
from tqdm import tqdm


def main(path, out_path, number_of_files):
    IN_FILENAME = "cstat{0}.pkl"
    OUT_FILENAME = "dstat{0}.pkl"
    total_out_files = len([f for f in Path(path).rglob("cstat*.pkl")])*number_of_files
    with tqdm(total=total_out_files) as pbar:
        for f in Path(path).rglob("cstat*.pkl"):
            with open(str(f), "rb") as fp:
                stats = pickle.load(fp)
            keys_per_file = len(stats) // number_of_files
            for i in range(number_of_files):
                partial_dict = dict(islice(stats.items(), i*keys_per_file, (i+1)*keys_per_file))
                with open(os.path.join(out_path, OUT_FILENAME.format(pbar.n)), "wb") as fp:
                    pickle.dump(partial_dict, fp)
                pbar.update(1)
                del partial_dict
            del stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('-out-path')
    parser.add_argument('-number_of_files', help='number of files to split the pgn into',
                        default=500, type=int)
    args = parser.parse_args()
    main(args.path, args.path if args.out_path is None else args.out_path, args.number_of_files)
