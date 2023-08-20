# example of training the resnet18 model
from __future__ import print_function, division

from typing import Dict
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm
from datetime import datetime
import git

from nn.pytorch_nn.AlphaChess import ValAccuracyBase
from nn.pytorch_nn.AlphaChess.ValAccuracyPolicy import AlphaValPolicy
from nn.pytorch_nn.AlphaChess.ValAccuracyValue import AlphaValValue
from nn.pytorch_nn.AlphaChess.utils import load_model, create_alpha_chess_model
from nn.pytorch_nn.data_loading.dataloaders import build_dataloaders

from shared.shared_functionality import get_config, value_to_outcome, \
    get_criterion


## Train model functions
def binary_accuracy(preds, labels):
    curr_acc = torch.sum(preds.round().long() == labels.data).double()
    res = (curr_acc / float(len(preds)))
    return res.item()


def train_helper(dataloaders, device, phase, optimizer, model, criterion, tensorboard, dataset_sizes, epoch,
                 writer=None):
    running_loss = 0.0

    last_loss = None
    # Iterate over data.
    start_time = time.time()
    for i, (inputs, labels) in tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase])):
        data_time = time.time() - start_time
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):

            outputs = model(inputs)

            # CrossEntropy loss is not symmetric, CrossEntropyLoss(B,A) = H(A, softmax(B)) where H is the Cross-Entropy function
            loss = criterion(outputs, labels.reshape(labels.shape[0], -1))

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()
            last_loss = loss
        # statistics
        running_loss += loss.item()  # * inputs.size(0)

        # del outputs, loss, labels
        # torch.cuda.empty_cache()
        train_time = time.time() - start_time - data_time
        if tensorboard == 'on':
            step = epoch * len(dataloaders[phase]) + i
            writer.add_scalar("Data Time", data_time, step)
            writer.add_scalar("Train Time", train_time, step)
        start_time = time.time()

    print(f'Last batch loss: {round(last_loss.item(), 5)}')
    epoch_loss = running_loss / len(dataloaders[phase])

    if tensorboard == 'on':
        writer.add_scalar("Train loss", epoch_loss, epoch)
        writer.add_scalar("Train lr", optimizer.param_groups[0]['lr'], epoch)

    print('Train loss: {:.4f}'.format(epoch_loss))


def train_helper_alpha_chess(dataloaders, device, phase, optimizer, model, criterion, tensorboard, dataset_sizes, epoch,
                             writer=None):
    running_loss: Dict[str, float] = {}
    epoch_loss: Dict[str, float] = {}

    for head in model.heads + ['tot']:
        running_loss[head] = 0.0
        epoch_loss[head] = 0.0

    last_loss = None
    # Iterate over data.
    start_time = time.time()
    for i, (inputs, labels_policy, labels_value, labels_masks) in tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase])):
        data_time = time.time() - start_time
        inputs = inputs.to(device)
        labels_policy = labels_policy.to(device)
        labels_value = labels_value.to(device)
        labels_masks = labels_masks.to(device)
        labels = {
            'policy_network': [labels_policy, labels_masks],
            'value_network': labels_value
        }
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):

            outputs = model(inputs)

            # CrossEntropy loss is not symmetric, CrossEntropyLoss(B,A) = H(A, softmax(B)) where H is the Cross-Entropy function
            loss, head_losses = criterion(outputs, labels)

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                optimizer.step()
            last_loss = loss
        # statistics
        for head in model.heads + ['tot']:
            running_loss[head] += head_losses[head].item()  # * inputs.size(0)

        # del outputs, loss, labels
        # torch.cuda.empty_cache()
        train_time = time.time() - start_time - data_time
        if tensorboard == 'on':
            step = epoch * len(dataloaders[phase]) + i
            writer.add_scalar("Data Time", data_time, step)
            writer.add_scalar("Train Time", train_time, step)
        start_time = time.time()

    print(f'Last batch loss: {round(last_loss.item(), 5)}')
    for head in ['tot'] + model.heads:
        epoch_loss[head] = running_loss[head] / len(dataloaders[phase])
        if tensorboard == 'on':
            writer.add_scalar(f"Train loss {head}", epoch_loss[head], epoch)
        print(f'Train loss {head}: {round(epoch_loss[head],4)}')

    if tensorboard == 'on':
        writer.add_scalar("Train lr", optimizer.param_groups[0]['lr'], epoch)




def val_alpha_chess_network(dataloaders, device, phase, model, criterion, tensorboard, writer, epoch):
    val_per_head_functions: Dict[str, ValAccuracyBase] = {
        'value_network': AlphaValValue,
        'policy_network': AlphaValPolicy,
    }

    alpha_val: Dict[str, ValAccuracyBase] = dict([(head, val_accuracy(writer, tensorboard, epoch)) for
                                                  head, val_accuracy in val_per_head_functions.items()
                                                  if head in model.heads])
    epoch_acc: Dict[str, float] = {}
    running_loss: Dict[str, float] = {}
    epoch_loss: Dict[str, float] = {}

    for head in model.heads + ['tot']:
        epoch_acc[head] = 0.0
        running_loss[head] = 0.0
    with torch.no_grad():
        for i, (inputs, labels_policy, labels_value, labels_masks) in tqdm(enumerate(dataloaders[phase]),
                                                             total=len(dataloaders[phase])):
            inputs = inputs.to(device)
            l_policy = labels_policy.numpy().reshape([labels_policy.shape[0], -1])
            l_masks = labels_masks.numpy().reshape([labels_masks.shape[0], -1])
            l_value = labels_value
            labels_policy = labels_policy.to(device)
            labels_value = labels_value.to(device)
            labels_masks = labels_masks.to(device)
            labels_cpu = {
                'policy_network': [l_policy, l_masks],
                'value_network': l_value,
            }
            labels = {
                'policy_network': [labels_policy, labels_masks],
                'value_network': labels_value
            }
            model(inputs)
            loss, head_losses = criterion(model, labels)
            running_loss['tot'] += loss.item()
            for head in model.heads:
                running_loss[head] += head_losses[head].item()
                # todo: apply torch.softmax on policy head
                o = model.head_outputs[head].detach().cpu().numpy()
                o = o.reshape([o.shape[0], -1])
                # calculate precision
                # we divide the output space into 3 categories: lose, draw, win
                alpha_val[head].update_accuracy_from_batch(labels_cpu[head], o)

    for head in ['tot'] + model.heads:
        if head in model.heads:
            epoch_acc[head] = alpha_val[head].print_and_log_accuracy()
        epoch_loss[head] = running_loss[head] / len(dataloaders[phase])
        print(f'Val loss {head}: {round(epoch_loss[head], 3)} ')
        if tensorboard == 'on':
            writer.add_scalar(f'Val Loss {head}', epoch_loss[head], epoch)
    return epoch_acc


def val_value_network(dataloaders, device, phase, model, criterion, tensorboard, writer, epoch):
    epoch_acc = 0.0
    tp = 0
    fp = 0
    running_loss = 0.0
    with torch.no_grad():
        for i, (inputs, labels) in tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase])):
            inputs = inputs.to(device)
            l = labels.numpy().reshape([labels.shape[0], -1])
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.reshape(labels.shape[0], -1))
            running_loss += loss.item()
            outputs = torch.clamp(outputs, -1.0, 1.0)
            o = outputs.detach().cpu().numpy()

            # calculate precision
            # we divide the output space into 3 categories: lose, draw, win
            l = value_to_outcome(l)
            o = value_to_outcome(o)
            tp += ((l == o).sum())
            fp += ((l != o).sum())
    epoch_acc = float(tp) / (tp + fp)
    epoch_loss = running_loss / len(dataloaders[phase])
    print(f'Val Precision: {round(epoch_acc, 3)} ')
    print(f'Val loss: {round(epoch_loss, 3)} ')
    if tensorboard == 'on':
        writer.add_scalar("Val Loss", epoch_loss, epoch)
        writer.add_scalar("Precision", epoch_acc, epoch)
    return epoch_acc


def val_policy_network(dataloaders, device, phase, model, criterion, tensorboard, writer, epoch, top_k=3):
    epoch_acc = 0.0
    tp = [0 for _ in range(top_k)]
    fp = [0 for _ in range(top_k)]

    running_loss = 0.0
    with torch.no_grad():
        for i, (inputs, labels) in tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase])):
            inputs = inputs.to(device)
            l = labels.numpy().reshape([labels.shape[0], -1])
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.reshape(labels.shape[0], -1))
            running_loss += loss.item()
            outputs = torch.softmax(outputs, dim=1)  # used with the CrossEntropy Loss
            o = outputs.detach().cpu().numpy()
            o_order = np.argsort(o, axis=1)
            for j in range(l.shape[0]):
                l_non_zero_indices = np.where(l[j] > 0)[0]
                for k in range(1, top_k + 1):
                    if set(o_order[j, -k:]).intersection(l_non_zero_indices):
                        tp[k - 1] += 1
                    else:
                        fp[k - 1] += 1

    if tensorboard == 'on':
        epoch_loss = running_loss / len(dataloaders[phase])
        writer.add_scalar("Val Loss", epoch_loss, epoch)
    print('Val Precision:')
    for k in range(1, top_k + 1):
        precision = float(tp[k - 1]) / (tp[k - 1] + fp[k - 1])
        if k == 1:
            epoch_acc = precision
        print(f'k : {k} precision: {round(precision, 3)} ')
        if tensorboard == 'on':
            writer.add_scalar(f"Val precision k={k}", precision, epoch)
    return epoch_acc


def set_model_train(model):
    if hasattr(model, 'heads'):
        model.set_train_mode()
    else:
        model.train()  # Set model to training mode


def set_model_eval(model):
    if hasattr(model, 'heads'):
        model.set_eval_mode()
    else:
        model.eval()  # Set model to training mode


def save_model(model, model_name, out_model_path, epoch, best_acc):
    model_filepath = str(
        out_model_path / f'model_{model_name}_epoch_{epoch}_acc_{best_acc:.4f}.pth'.format(**vars()))
    print('saving model: ', model_filepath)
    torch.save(model.state_dict(), model_filepath)
    if hasattr(model, 'head_networks'):
        for head in model.head_networks:
            model_filepath = str(
                out_model_path / f'model_{model_name}_epoch_{epoch}_acc_{best_acc:.4f}_{head}.pth'.format(**vars()))
            torch.save(model.head_networks[head].state_dict(), model_filepath)


def train(model, criterion, optimizer, lr_scheduler, dataloaders, device, dataset_sizes, num_epochs, out_model_path,
          model_name, tensorboard, writer):
    since = time.time()
    best_acc = 0.0
    config = get_config()
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        os.makedirs(out_model_path, exist_ok=True)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':

                set_model_train(model)
                train_network = get_train_network(config)
                train_network(dataloaders, device, phase, optimizer, model, criterion, tensorboard, dataset_sizes,
                              epoch,
                              writer=writer)

            elif phase == 'val':
                with torch.no_grad():
                    set_model_eval(model)
                val_network = get_val_network(config)
                epoch_acc = val_network(dataloaders, device, phase, model, criterion, tensorboard, writer, epoch)
                epoch_acc = epoch_acc['value_network']
                lr_scheduler.step()
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    save_model(model, model_name, out_model_path, epoch, best_acc)

        epoch_time = time.time() - epoch_start
        print('Epoch {:.0f}m {:.0f}s'.format(
            (epoch_time) // 60, epoch_time % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


def get_val_network(config):
    network_name = config['train']['torch']['network_name']
    networks = {
        "ValueNetwork": val_value_network,
        "PolicyNetwork": val_policy_network,
        "AlphaChessNetwork": val_alpha_chess_network
    }
    if network_name in networks:
        return networks[network_name]
    raise Exception("network name not available")


def get_train_network(config):
    network_name = config['train']['torch']['network_name']
    networks = {
        "ValueNetwork": train_helper,
        "PolicyNetwork": train_helper,
        "AlphaChessNetwork": train_helper_alpha_chess
    }
    if network_name in networks:
        return networks[network_name]
    raise Exception("network name not available")


def save_git_commit(writer: SummaryWriter) -> None:
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    message = repo.head.object.message
    writer.add_text('git_sha', sha)
    writer.add_text('git_message', message)

def main():
    # call args in args
    out_model_path = args.out_model_path
    model_name = args.model_name
    data_dir = args.data_dir
    tensorboard = args.tensorboard
    log_path = args.log_path
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    step_size = args.step_size
    gamma = args.gamma
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    verbose = args.verbose
    in_model_path = args.in_model_path
    freeze_body = args. freeze_body
    freeze_value = args.freeze_value
    freeze_policy = args.freeze_policy

    now = datetime.now()
    out_model_path = Path(out_model_path) / now.strftime("%Y_%m_%d___%H_%M_%S")

    # start writer for tensorboard (if 'on')
    writer = None
    if tensorboard == 'on':
        writer = SummaryWriter(log_path)
        save_git_commit(writer)

    # build dataset, load it onto dataloader, and get relevant information (dataset size, class names)
    is_local = torch.cuda.device_count() == 1  # todo: use data parallel
    loader_name = "base_loader_params"
    if not is_local:
        loader_name = "strong_loader_params"

    used_split_types = ['train', 'val']
    dataloaders = build_dataloaders(data_dir, loader_name, used_split_types, verbose)
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in used_split_types}

    # set device to be either GPU (if available) or CPU (if GPU not available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = get_config()
    criterion = get_criterion(config).to(device)

    model = create_alpha_chess_model(device, freeze_body, freeze_policy, freeze_value)
    if in_model_path is not None:
        load_model(model, in_model_path, config)

        print('checking precision of loaded model:')
        with torch.no_grad():
            val_network = get_val_network(config)
            val_network(dataloaders, device, 'val', model, criterion, tensorboard, writer, 0)
        print('continuing to train')

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr=learning_rate,
                              weight_decay=weight_decay)  # add weight decay here, if needed

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)

    # Train model
    model = train(model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, device, dataset_sizes,
                  num_epochs=num_epochs, out_model_path=out_model_path, model_name=model_name, tensorboard=tensorboard,
                  writer=writer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--out_model_path', type=str, default='/home/matan/models/my_models/chess',
                        help='location to save models')
    parser.add_argument('--in_model_path', type=str, default=None, help='location of model start train from')
    parser.add_argument('--model_name', type=str,
                        default='10_09_22_exp1',
                        help='name of model to be used')
    parser.add_argument('--freeze_body', action='store_true')
    parser.add_argument('--freeze_value', action='store_true')
    parser.add_argument('--freeze_policy', action='store_true')
    parser.add_argument('--data_dir', type=str, default='/home/matan/data/mydata/chess/caissabase/pgn/mstat_100',
                        help='location of folder of images to be trained and validated')
    parser.add_argument('--tensorboard', type=str, default='on', help='start loss/acc tracking using tensorboard')
    parser.add_argument('--log_path', type=str, default='runs/chess/val_logs', help='folder of tensorboard logs')
    parser.add_argument('--learning_rate', type=int, default=0.001, help='value of learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='step size of epochs for learning decay')
    parser.add_argument('--step_size', type=int, default=10, help='step size of epochs for learning decay')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor')
    parser.add_argument('--batch_size', type=int, default=20, help='recommended>=32 and not too big.')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    main()
