import json
import torch

from my_chess.shared.shared_functionality import get_config_path
from my_chess.nn.pytorch_nn.estat_dataset import Estat_Dataset


def loader(f):
    return f

def build_datasets(data_dir, loader_name, verbose=False):
    config_path = get_config_path()
    with open(config_path) as fp:
        config = json.load(fp)
    data = config['train']['torch']['data_partitioning']
    loader_params = config['train']['torch']['data_loader']['base_loader_params']
    loader_params.update(config['train']['torch']['data_loader'][loader_name])
    dl = {}
    for k in data:
        dl[k] = torch.utils.data.DataLoader(Estat_Dataset(k, data_dir,
                                                          shuffle=loader_params['shuffle'],
                                                          num_workers=loader_params['num_workers'],
                                                          pin_memory=loader_params['pin_memory']))
    return dl

if __name__ == '__main__':
    # folder_stats('/Users/assaflehr/datasets/trolly_crops','val')
    ds = build_datasets('/home/matan/data/mydata/chess/caissabase/pgn/estat_small')['train']
    # ds = build_datasets('/home/matan/rep/flip_camera_detector/flip_camera_dataset')['val']
    print(len(ds))

