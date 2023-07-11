import json
import numpy as np
import torch

from shared.shared_functionality import get_config_path
from nn.pytorch_nn.estat_dataset import Estat_Dataset
from shared.shared_functionality import INPUT_PLANES, OUTPUT_PLANES


def collate_fn(data):
    in_size = 0
    out_size = 0
    for item in data:
        in_size += item['in'].shape[0]
        out_size += item['out'].shape[0]
    result = {
        'in': torch.zeros([in_size, 8, 8, INPUT_PLANES], dtype=torch.float32),
        'out': torch.zeros([out_size, 8, 8, OUTPUT_PLANES], dtype=torch.float32)
    }
    in_index = 0
    out_index = 0
    for item in data:
        result['in'][in_index:in_index + item['in'].shape[0]] = torch.Tensor(item['in'])
        result['out'][out_index:out_index + item['out'].shape[0]] =  torch.Tensor(item['out'])
        in_index += item['in'].shape[0]
        out_index += item['out'].shape[0]
    return result['in'], result['out']


def build_dataloaders(data_dir, loader_name, used_split_types, verbose=False):
    config_path = get_config_path()
    with open(config_path) as fp:
        config = json.load(fp)
    loader_params = config['train']['torch']['data_loader']['base_loader_params']
    loader_params.update(config['train']['torch']['data_loader'][loader_name])
    print(f'building dataset with params: {json.dumps(loader_params, indent=4)}')
    dl = {}
    for k in used_split_types:
        dl[k] = torch.utils.data.DataLoader(Estat_Dataset(k, data_dir),
                                            batch_size=loader_params['batch_size'],
                                            shuffle=loader_params['shuffle'],
                                            num_workers=loader_params['num_workers'],
                                            pin_memory=loader_params['pin_memory'],
                                            collate_fn=collate_fn
                                            )
    return dl


if __name__ == '__main__':
    ds = build_dataloaders('/home/matan/data/mydata/chess/caissabase/pgn/estat', 'base_loader_params')['train']
    print(len(ds.dataset))
