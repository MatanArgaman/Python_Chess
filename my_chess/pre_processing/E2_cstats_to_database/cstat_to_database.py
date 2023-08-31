# splits cstat file by hash384 to create smaller files which can be accessed quicker
# and used as a database

import pickle
from itertools import islice
from pathlib import Path
import os
from tqdm import tqdm
import argparse
import re

from shared.shared_functionality import board_fen_to_hash384


def main(path, out_path, number_of_files):
    IN_FILENAME = "cstat{0}.pkl"
    OUT_FILENAME = "database{0}_{1}.pkl"
    total_out_files = len([f for f in Path(path).rglob("cstat*.pkl")])*number_of_files
    with tqdm(total=total_out_files) as pbar:
        for f in Path(path).rglob("cstat*.pkl"):
            m = re.search("cstat([0-9]*).pkl", os.path.basename(f)) # hash 256
            index1 = m.group(1)
            with open(str(f), "rb") as fp:
                cstats = pickle.load(fp)

            database = {}
            for k,v in cstats.items():
                index2 = board_fen_to_hash384(k) % number_of_files
                db_per_index2 = database.get(index2, {})
                database[index2] = db_per_index2
                db_per_index2[k] = v
            for index2, db_per_index2 in database.items():
                with open(os.path.join(out_path, OUT_FILENAME.format(index1, index2)), "wb") as fp:
                    pickle.dump(db_per_index2, fp)
                    pbar.update(1)
            del cstats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('out_path')
    parser.add_argument('-number_of_files', help='number of files to split the pgn into',
                        default=500, type=int)
    args = parser.parse_args()
    main(args.path, args.path if args.out_path is None else args.out_path, args.number_of_files)
