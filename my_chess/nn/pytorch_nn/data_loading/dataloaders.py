import json
import torch

from shared.shared_functionality import get_config_path, get_dataloader, get_collate_function
from shared.shared_functionality import INPUT_PLANES, OUTPUT_PLANES


def build_dataloaders(data_dir, loader_name, used_split_types, verbose=False):
    config_path = get_config_path()
    with open(config_path) as fp:
        config = json.load(fp)
    loader_params = config['train']['torch']['data_loader']['base_loader_params']
    loader_params.update(config['train']['torch']['data_loader'][loader_name])
    print(f'building dataset with params: {json.dumps(loader_params, indent=4)}')
    dl = {}
    dataloader = get_dataloader(config)
    collate_function = get_collate_function(config)
    for k in used_split_types:
        dl[k] = torch.utils.data.DataLoader(dataloader(k, data_dir, shuffle=loader_params['shuffle']),
                                            batch_size=loader_params['batch_size'],
                                            shuffle= False, # leave this as False or there will always be cache misses, shuffle is done inside DataLoader - per file
                                            num_workers=loader_params['num_workers'],
                                            pin_memory=loader_params['pin_memory'],
                                            collate_fn = collate_function,
                                            drop_last=True
                                            )
    return dl


if __name__ == '__main__':
    ds = build_dataloaders('/home/matan/data/mydata/chess/caissabase/pgn/estat', 'base_loader_params')['train']
    print(len(ds.dataset))
