from typing import List, Dict

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
        self.block = BasicAlphaChessBlock(256, 256, 1)
        self.conv1 = conv3x3(INPUT_PLANES, 256, 1)
        self.bn1 = nn.BatchNorm2d(256, eps=1e-05)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        for _ in range(19):
            x = self.block(x)
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
        x = self.tanh(x)
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
        self.body = AlphaChessBody()
        self.heads = heads
        self.head_weights = head_weights

    def forward(self, x):
        x = self.body(x)
        for head in self.heads:
            if head in self.head_weights and self.head_weights[head] != 0:
                setattr(self, head, self.head_dict[head]()(x))
            else:
                LOG.warning(f'dropping head {head}')
        return self


class AlphaChessLoss(nn.Module):
    def __init__(self, heads: List[str], head_weights: Dict[str, int]):
        super(AlphaChessLoss, self).__init__()
        self.heads = heads
        self.head_weights = head_weights
        self.loss_dict = {
            'value_network': nn.MSELoss(reduction='mean'),
            'policy_network': nn.CrossEntropyLoss(reduction='mean')
        }

    def forward(self, model, labels):
        losses = {head: 0 for head in self.heads}
        d = model.__dict__
        losses['tot'] = 0
        for head in self.heads:
            if head in d:
                losses[head] = self.loss_dict[head](d[head], labels)
                losses['tot'] += self.head_weights[head] * losses[head]
        return losses['tot']  # , losses


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
    loss = AlphaChessLoss(heads, head_weights)
    return loss
