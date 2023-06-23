import os.path

from torch.utils.data import Dataset
from scipy.sparse import load_npz
from pathlib import Path
import re
import numpy as np
from torchvision import transforms
import json

from my_chess.shared.shared_functionality import get_config_path
from my_chess.shared.shared_functionality import INPUT_PLANES, OUTPUT_PLANES


def load_files(root_dir):
    estat_in = {}
    estat_out = {}
    for f in Path(root_dir).rglob('*.npz'):
        f = str(f)
        m = re.search('estat([0-9]*)_o.npz', f)
        if m:
            estat_out[m.group(1)] = f

        m = re.search('estat([0-9]*)_i.npz', f)
        if m:
            estat_in[m.group(1)] = f

    s_in = set(estat_in.keys())
    s_out = set(estat_out.keys())
    for k in s_out.difference(s_in):
        del [estat_out[k]]
    for k in s_in.difference(s_out):
        del [estat_in[k]]
    assert set(estat_in.keys()).__eq__(set(estat_out.keys())), \
        "difference in keys is not expected after the previous lines which were supposed to remove such keys."

    indices = list(estat_in.keys())
    return indices, estat_in, estat_out


class Estat_Dataset(Dataset):
    def __init__(self, split_type, root_dir, seed=5):
        np.random.seed(seed)

        self._split_type = split_type
        self.root_dir = root_dir
        indices, self.estat_in, self.estat_out = load_files(root_dir)

        config_path = get_config_path()
        with open(config_path) as fp:
            config = json.load(fp)
        out_filename = config['train']['torch']['data_partition_filename']

        partition_path = os.path.join(root_dir, out_filename)
        with open(partition_path) as fp:
            partition = json.load(fp)
        self.indices = indices[indices == partition[split_type]]

    def __len__(self):
        return len(self.estat_in)

    def __getitem__(self, idx):
        index = self.indices[idx]
        sample_in = load_npz(self.estat_in[index]).toarray()
        sample_in = sample_in.reshape(sample_in.shape[0], 8, 8, INPUT_PLANES)

        sample_out = load_npz(self.estat_out[index]).toarray()
        sample_out = sample_out.reshape(sample_out.shape[0], 8, 8, OUTPUT_PLANES)
        result = {
            'in': sample_in,
            'out': sample_out
        }
        return result
