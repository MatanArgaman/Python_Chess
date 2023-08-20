from __future__ import division, print_function

import os
from typing import Dict

import torch

from shared.shared_functionality import get_config, get_model, data_parallel, freeze_layers


def get_state_dict(in_model_path: str) -> Dict:
    device_count = torch.cuda.device_count()
    state_dict = torch.load(in_model_path)
    if device_count <= 1:
        new_state_dict = {}
        # remove module prefix since not using data_parallel
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        state_dict = new_state_dict
    return state_dict


def load_model(model, in_model_path, config):
    model.load_state_dict(get_state_dict(in_model_path))
    for head in config['train']['torch']['network_heads']:
        head_path = in_model_path[:-4] + f'_{head}' + in_model_path[-4:]
        if os.path.exists(head_path):
            model.head_networks[head].load_state_dict(get_state_dict(head_path))
        else:
            print(f'skipping loading of head {head} weights')


def create_alpha_chess_model(device, freeze_body, freeze_policy, freeze_value):
    config = get_config()
    model = get_model(config)
    model.body = data_parallel(model.body).to(device)
    for head in model.heads:
        model.head_networks[head] = data_parallel(model.head_networks[head]).to(device)
    freeze_model_networks(model, freeze_body, freeze_policy, freeze_value)
    return model


def freeze_model_networks(model, freeze_body, freeze_policy, freeze_value):
    if freeze_body:
        freeze_layers(model.body)
    if freeze_policy:
        freeze_layers(model.head_networks['policy_network'])
    if freeze_value:
        freeze_layers(model.head_networks['value_network'])
