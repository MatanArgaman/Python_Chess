import os.path

from torch.utils.data import Dataset
from scipy.sparse import load_npz
from pathlib import Path
import re
import numpy as np
import json
import pickle

from shared.shared_functionality import get_config_path
from shared.shared_functionality import INPUT_PLANES, OUTPUT_PLANES


def load_files(root_dir):
    estat_in = {}
    estat_out = {}
    estat_size = set()
    for f in Path(root_dir).rglob('*.npz'):
        f = str(f)
        m = re.search('estat([0-9]*)_o.npz', f)
        if m:
            estat_out[m.group(1)] = f

        m = re.search('estat([0-9]*)_i.npz', f)
        if m:
            estat_in[m.group(1)] = f
    for f in Path(root_dir).rglob('*.pkl'):
        f = str(f)
        m = re.search('estat([0-9]*)_s.pkl', f)
        if m:
            estat_size.add(m.group(1))

    s_in = set(estat_in.keys())
    s_out = set(estat_out.keys())
    for k in s_out.difference(s_in):
        del estat_out[k]
        s_out.remove(k)
    for k in s_in.difference(s_out):
        del estat_in[k]
        s_in.remove(k)
    for k in s_in.difference(estat_size):
        del estat_in[k]
        del estat_out[k]

    assert set(estat_in.keys()).__eq__(set(estat_out.keys())), \
        "difference in keys is not expected after the previous lines which were supposed to remove such keys."

    indices = np.array(list(estat_in.keys()))
    return indices, estat_in, estat_out


class Estat_Dataset(Dataset):
    def __init__(self, split_type, root_dir, seed=5, file_cache_max_size=2):
        np.random.seed(seed)
        self._file_cache_max_size = file_cache_max_size
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
        self.estat_indices = indices[[x in partition[split_type] for x in indices]]
        self.estat_indices = indices[np.random.permutation(self.estat_indices.size)]

        self.files_sample_size_accumulative = [0]
        for file_index in self.estat_indices:
            path = self.estat_in[file_index].replace('_i', '_s').replace('.npz', '.pkl')
            with open(path, 'rb') as fp:
                samples = pickle.load(fp)
            self.files_sample_size_accumulative.append(self.files_sample_size_accumulative[-1] + samples)
        self.files_sample_size_accumulative = np.array(self.files_sample_size_accumulative, dtype=int)
        self.file_cache = {
            'samples': {},
            'last_used_file_indices': []
        }
        # samples - keyL file_index, value: {'in': in_npz, 'out': out_npz}
        # last_used_file_indices - list of file_index where self.file_cache['last_used_file_indices'][0] is the least recently used

    def __len__(self):
        return self.files_sample_size_accumulative[-1]

    def __getitem__(self, idx):
        sample_accumulative_index = np.searchsorted(self.files_sample_size_accumulative, idx, side='left') - 1
        if idx in self.files_sample_size_accumulative:
            sample_accumulative_index += 1
        file_index = self.estat_indices[sample_accumulative_index]

        if file_index not in self.file_cache['samples']:
            if len(self.file_cache['last_used_file_indices']) >= self._file_cache_max_size:
                removed_file_index = self.file_cache['last_used_file_indices'].pop(0)
                del self.file_cache['samples'][removed_file_index]

            sample_in = load_npz(self.estat_in[file_index]).toarray()
            sample_in = sample_in.reshape(sample_in.shape[0], 8, 8, INPUT_PLANES)
            sample_out = load_npz(self.estat_out[file_index]).toarray()
            sample_out = sample_out.reshape(sample_out.shape[0], 8, 8, OUTPUT_PLANES)
            self.file_cache['samples'][file_index] = {
                'in': sample_in,
                'out': sample_out
            }
            self.file_cache['last_used_file_indices'].append(file_index)

        sample_in = self.file_cache['samples'][file_index]['in']
        sample_out = self.file_cache['samples'][file_index]['out']

        in_file_idx = idx - self.files_sample_size_accumulative[sample_accumulative_index]

        result = {
            'in': sample_in[[in_file_idx]],
            'out': sample_out[[in_file_idx]]
        }
        return result
