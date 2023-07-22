# example of training the resnet18 model
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import copy
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np

from nn.pytorch_nn.dataloaders import build_dataloaders
from resnet import MyResNet18
import argparse
from tqdm import tqdm

from shared.shared_functionality import data_parallel


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


def val_helper(dataloaders, device, phase, model, criterion, tensorboard, writer, epoch, top_k=3):
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


def train(model, criterion, optimizer, lr_scheduler, dataloaders, device, dataset_sizes, num_epochs, model_path,
          model_name, tensorboard, writer):
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        folder = Path(model_path) / model_name
        os.makedirs(folder, exist_ok=True)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                train_helper(dataloaders, device, phase, optimizer, model, criterion, tensorboard, dataset_sizes, epoch,
                             writer=writer)

            elif phase == 'val':
                with torch.no_grad():
                    model.eval()  # Set model to evaluate mode
                epoch_acc = val_helper(dataloaders, device, phase, model, criterion, tensorboard, writer, epoch)
                lr_scheduler.step()
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    model_filepath = str(folder / f'model_{model_name}_epoch_{epoch}_acc_{best_acc:.4f}.pth'.format(**vars()))
                    print('saving model: ', model_filepath)
                    torch.save(model.state_dict(), model_filepath)

        epoch_time = time.time() - epoch_start
        print('Epoch {:.0f}m {:.0f}s'.format(
            (epoch_time) // 60, epoch_time % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


def main():
    # call args in args
    model_path = args.model_path
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
    load_model = args.load_model

    # start writer for tensorboard (if 'on')
    writer = None
    if tensorboard == 'on':
        writer = SummaryWriter(log_path)

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

    # create a resnet18 instance and load it onto device
    model = data_parallel(MyResNet18()).to(device)

    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)

    if load_model is not None:
        model.load_state_dict(torch.load(load_model))
        print('checking precision of loaded model:')
        with torch.no_grad():
            val_helper(dataloaders, device, 'val', model, criterion, tensorboard, writer, 0)
        print('continuing to train')

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr=learning_rate,
                              weight_decay=weight_decay)  # add weight decay here, if needed

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)

    # Train model
    model = train(model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, device, dataset_sizes,
                  num_epochs=num_epochs, model_path=model_path, model_name=model_name, tensorboard=tensorboard,
                  writer=writer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str, default='/home/matan/models/my_models/chess',
                        help='location to save models')
    parser.add_argument('--load_model', type=str, default=None, help='location of model start train from')
    parser.add_argument('--model_name', type=str,
                        default='10_09_22_exp1',
                        help='name of model to be used')

    parser.add_argument('--data_dir', type=str, default='/home/matan/data/mydata/chess/caissabase/pgn/estat_100',
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
