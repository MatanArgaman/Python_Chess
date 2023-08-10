from typing import List, Dict

import torch
import torch.nn as nn

from shared.shared_functionality import OUTPUT_PLANES, INPUT_PLANES, SingletonLogger, get_config

LOG = SingletonLogger().get_logger('train')


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


def conv1x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=True)


class BasicAlphaChessBlock(nn.Module):
    def __init__(self, in_planes, planes, stride):
        super(BasicAlphaChessBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-05)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class AlphaChessBody(nn.Module):
    def __init__(self):
        super(AlphaChessBody, self).__init__()
        self.blocks = nn.Sequential(*[BasicAlphaChessBlock(256, 256, 1) for _ in range(19)])
        self.conv1 = conv3x3(INPUT_PLANES, 256, 1)
        self.bn1 = nn.BatchNorm2d(256, eps=1e-05)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.blocks(x)
        return x


class AlphaChessHeadValue(nn.Module):
    def __init__(self):
        super(AlphaChessHeadValue, self).__init__()
        self.conv1 = conv1x1(256, 1, 1)
        self.bn1 = nn.BatchNorm2d(1, eps=1e-05)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(64, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        # x = self.tanh(x)
        return x


class AlphaChessHeadPolicy(nn.Module):
    def __init__(self):
        super(AlphaChessHeadPolicy, self).__init__()
        self.conv1 = conv3x3(256, 256, 1)
        self.bn1 = nn.BatchNorm2d(256, eps=1e-05)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(256, OUTPUT_PLANES, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class AlphaChessComplete(nn.Module):
    def __init__(self, heads: List[str], head_weights: Dict[str, int]):
        super(AlphaChessComplete, self).__init__()
        self.head_dict = {
            'policy_network': AlphaChessHeadPolicy,
            'value_network': AlphaChessHeadValue
        }
        self.body: nn.Module = AlphaChessBody()
        self.heads: List[str] = heads
        self.head_weights: Dict[str, int] = head_weights
        self.head_outputs: Dict[str, torch.Tensor] = {}
        self.head_networks: Dict[str, nn.Module] = {}
        for head in self.heads:
            if head in self.head_weights and self.head_weights[head] != 0:
                self.head_networks[head] = self.head_dict[head]()

    def set_train_mode(self):
        self.train()
        self.body.train()
        for head in self.heads:
            self.head_networks[head].train()

    def set_eval_mode(self):
        self.eval()
        self.body.eval()
        for head in self.heads:
            self.head_networks[head].eval()

    def forward(self, x):
        x = self.body(x)
        for head in self.heads:
            if head in self.head_weights and self.head_weights[head] != 0:
                self.head_outputs[head] = self.head_networks[head](x)
            else:
                LOG.warning(f'dropping head {head}')
        return self


class AlphaChessLoss(nn.Module):
    def __init__(self, heads: List[str], head_weights: Dict[str, int], policy_loss_move_weight: float):
        super(AlphaChessLoss, self).__init__()
        self.heads = heads
        self.head_weights = head_weights
        self.policy_loss_move_weight = policy_loss_move_weight
        self.loss_dict = {
            'value_network': nn.MSELoss(reduction='mean'),
            'policy_network': nn.CrossEntropyLoss(reduction='none')
        }

    def forward(self, model, labels):
        losses = {head: 0 for head in self.heads}
        losses['tot'] = 0
        for head in self.heads:
            if head in model.head_networks:
                if head == 'policy_network':
                    policy_labels = labels[head][0]
                    mask_weights = labels[head][1]
                    # threshold just to differentiate between 0 and 1s (as we're dealing with floats)
                    mask_weights[mask_weights < 0.5] = self.policy_loss_move_weight
                    losses[head] = torch.mean(mask_weights *
                                              self.loss_dict[head](model.head_outputs[head].view(model.head_outputs[head].shape[0], -1),
                                                                   policy_labels.view(model.head_outputs[head].shape[0], -1)))
                else:
                    losses[head] = self.loss_dict[head](model.head_outputs[head],
                                                        labels[head].view(model.head_outputs[head].shape))
                losses['tot'] += self.head_weights[head] * losses[head]
        return losses['tot'], losses  # , losses


def get_alpha_chess_model():
    config = get_config()
    train = config['train']['torch']
    heads = train['network_heads']
    head_weights = train['head_weights']
    model = AlphaChessComplete(heads, head_weights)
    return model


def get_alpha_chess_losses():
    config = get_config()
    train = config['train']['torch']
    heads = train['network_heads']
    head_weights = train['head_weights']
    loss = AlphaChessLoss(heads, head_weights, train['policy_loss_move_weight'])
    return loss
