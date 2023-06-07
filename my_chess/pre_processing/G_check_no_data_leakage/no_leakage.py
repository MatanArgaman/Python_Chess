import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.sparse import load_npz
import os
import json

from shared.shared_functionality import get_config_path, get_nn_io_file, get_all_train_files_indices
from shared.shared_functionality import INPUT_PLANES

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=get_config_path(), help='configuration file path')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    con_train = config['train']
    file_indices = get_all_train_files_indices(config)
    ex = set()
    for i in tqdm(file_indices):
        x_train = get_nn_io_file(i,con_train, is_input=True)
        x_train = x_train.toarray().reshape([8, 8, -1, INPUT_PLANES]).swapaxes(0, 2).swapaxes(1, 2)
        for j in range(x_train.shape[0]):
            key = x_train[j].tostring()
            if key in ex:
                print("data leakage")
            ex.add(key)

