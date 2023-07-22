import argparse
import json
import os
import numpy as np

from shared.shared_functionality import get_config_path

def main(in_dir, is_vstats=False, seed = 5):
    np.random.seed(seed)
    if is_vstats:
        from nn.pytorch_nn.vstat_dataset import load_files
    else:
        from nn.pytorch_nn.estat_dataset import load_files

    indices, stat_in, stat_out = load_files(in_dir)
    indices = list(indices[np.random.permutation(indices.size)])
    config_path = get_config_path()
    with open(config_path) as fp:
        config = json.load(fp)
    data = config['train']['torch']['data_partitioning']
    out_filename = config['train']['torch']['data_partition_filename']
    out_path = os.path.join(in_dir, out_filename)
    percentage = 0
    total = 0
    partition_item_count = []
    split_types = []
    for i, (k, v) in enumerate(data.items()):
        split_types.append(k)
        percentage += v
        if i == len(data) - 1:
            remaining = len(stat_in) - total
            partition_item_count.append(remaining)
        else:
            n = int(len(stat_in) * (float(v) / 100))
            total += n
            partition_item_count.append(n)
    assert percentage == 100, "percentage percentage should be 100"
    assert np.sum(partition_item_count) == len(stat_in)

    partition = {}
    total = 0
    for s, p in zip(split_types, partition_item_count):
        partition[s] = indices[total:total + p]
        total+=p

    with open(out_path, 'w') as fp:
        json.dump(partition,fp, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir')
    parser.add_argument('--is_vstats', action='store_true')
    args = parser.parse_args()
    main(args.in_dir, args.is_vstats)
