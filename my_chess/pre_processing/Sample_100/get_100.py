import argparse
import pickle
from scipy.sparse import load_npz, save_npz
from pathlib import Path
import re
import os

def main(source, dest):
    file_indices = []

    for f in Path(source).rglob('mstat*_i.npz'):
        f = str(f)
        m = re.search('mstat([0-9]*)_i.npz', f)
        file_indices.append(m.group(1))

    file_indices = list(set(file_indices)) # make unique
    for index in file_indices:
        d = os.path.join(dest, f'mstat{index}_s.pkl')
        with open(d, 'wb') as fp:
            pickle.dump(int(100), fp)

        s = os.path.join(source, f'mstat{index}_v.pkl')
        d = os.path.join(dest, f'mstat{index}_v.pkl')
        with open(s, 'rb') as fp:
            data = pickle.load(fp)
        with open(d, 'wb') as fp:
            pickle.dump(data[:100], fp)

        s = os.path.join(source, f'mstat{index}_m.pkl')
        d = os.path.join(dest, f'mstat{index}_m.pkl')
        with open(s, 'rb') as fp:
            data = pickle.load(fp)
        with open(d, 'wb') as fp:
            pickle.dump(data[:100], fp)

        s = os.path.join(source, f'mstat{index}_i.npz')
        d = os.path.join(dest, f'mstat{index}_i.npz')
        data = load_npz(s)
        save_npz(d, data[:100])

        s = os.path.join(source, f'mstat{index}_o.npz')
        d = os.path.join(dest, f'mstat{index}_o.npz')
        data = load_npz(s)
        save_npz(d, data[:100])





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('dest')
    args = parser.parse_args()
    main(args.source, args.dest)