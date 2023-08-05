import os.path

from torch.utils.data import Dataset
from scipy.sparse import load_npz
from pathlib import Path
import re
import numpy as np
import json
import pickle

from shared.shared_functionality import get_config
from shared.shared_functionality import INPUT_PLANES, OUTPUT_PLANES


def load_files(root_dir):
    mstat_in = {}
    mstat_out_policy = {}
    mstat_out_value = {}
    mstat_out_masks = {}
    mstat_size = set()
    for f in Path(root_dir).rglob('*.npz'):
        f = str(f)
        m = re.search('mstat([0-9]*)_o.npz', f)
        if m:
            mstat_out_policy[m.group(1)] = f

        m = re.search('mstat([0-9]*)_i.npz', f)
        if m:
            mstat_in[m.group(1)] = f

    for f in Path(root_dir).rglob('*.pkl'):
        f = str(f)
        m = re.search('mstat([0-9]*)_s.pkl', f)
        if m:
            mstat_size.add(m.group(1))
        m = re.search('mstat([0-9]*)_v.pkl', f)
        if m:
            mstat_out_value[m.group(1)] = f
        m = re.search('mstat([0-9]*)_m.pkl', f)
        if m:
            mstat_out_masks[m.group(1)] = f



    l = [mstat_in, mstat_out_policy, mstat_out_value, mstat_out_masks]
    s = set(l[0])
    for item in l:
        s = s.intersection(set(item))
    for item in l:
        diff_set = set(item).difference(s)
        for k in diff_set:
            del item[k]

    assert set(mstat_in.keys()).__eq__(set(mstat_out_masks.keys())), \
        "difference in keys is not expected after the previous lines which were supposed to remove such keys."
    assert set(mstat_in.keys()).__eq__(set(mstat_out_policy.keys())), \
        "difference in keys is not expected after the previous lines which were supposed to remove such keys."
    assert set(mstat_in.keys()).__eq__(set(mstat_out_value.keys())), \
        "difference in keys is not expected after the previous lines which were supposed to remove such keys."

    indices = np.array(list(mstat_in.keys()))
    return indices, mstat_in, mstat_out_policy, mstat_out_value


class MstatDataset(Dataset):
    def __init__(self, split_type, root_dir, seed=5, file_cache_max_size=1, shuffle=False):
        np.random.seed(seed)
        self._file_cache_max_size = file_cache_max_size
        self._split_type = split_type
        self._shuffle = shuffle
        self.root_dir = root_dir
        indices, self.mstat_in, self.mstat_out_policy, self.mstat_out_value = load_files(root_dir)

        config = get_config()
        out_filename = config['train']['torch']['data_partition_filename']

        partition_path = os.path.join(root_dir, out_filename)
        with open(partition_path) as fp:
            partition = json.load(fp)
        self.mstat_indices = indices[[x in partition[split_type] for x in indices]]
        self.mstat_indices = self.mstat_indices[np.random.permutation(self.mstat_indices.size)]

        self.files_sample_size_accumulative = [0]
        for file_index in self.mstat_indices:
            path = self.mstat_in[file_index].replace('_i', '_s').replace('.npz', '.pkl')
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
        file_index = self.mstat_indices[sample_accumulative_index]

        if file_index not in self.file_cache['samples']:
            if len(self.file_cache['last_used_file_indices']) >= self._file_cache_max_size:
                removed_file_index = self.file_cache['last_used_file_indices'].pop(0)
                del self.file_cache['samples'][removed_file_index]

            sample_in = load_npz(self.mstat_in[file_index]).toarray()
            sample_in = sample_in.reshape(sample_in.shape[0], 8, 8, INPUT_PLANES)
            sample_out = np.load(self.mstat_out[file_index])
            sample_out = sample_out.reshape(sample_out.shape[0], 8, 8, OUTPUT_PLANES)
            self.file_cache['samples'][file_index] = {
                'in': sample_in,
                'out': sample_out
            }
            if self._shuffle: # shuffles the samples within a file (not within the entire dataset)
                self.file_cache['samples'][file_index]['permutation'] = np.random.permutation(sample_in.shape[0])
            self.file_cache['last_used_file_indices'].append(file_index)

        sample_in = self.file_cache['samples'][file_index]['in']
        sample_out = self.file_cache['samples'][file_index]['out']

        in_file_idx = idx - self.files_sample_size_accumulative[sample_accumulative_index]
        if self._shuffle:
            in_file_idx = self.file_cache['samples'][file_index]['permutation'][in_file_idx]

        # swap axes explanation:
        # nn.conv2d filter works expected batch_size, in_planes, ....
        # since out data is current [batch_size, 8,8, IN_PLANES] we want to switch order so the convolutions will
        # work on a a [8, 8] array and not [8 , IN_PLANES] array.
        result = {
            'in': sample_in[[in_file_idx]].swapaxes(1,3).swapaxes(2,3),
            'out': sample_out[[in_file_idx]]
        }
        return result
