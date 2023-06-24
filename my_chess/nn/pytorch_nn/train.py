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

from nn.pytorch_nn.dataloaders import build_dataloaders
from resnet import MyResNet18
import argparse
from tqdm import tqdm


## Train model functions
def binary_accuracy(preds, labels):
    curr_acc = torch.sum(preds.round().long() == labels.data).double()
    res = (curr_acc / float(len(preds)))
    return res.item()


def train(model, criterion, optimizer, lr_scheduler, dataloaders, device, dataset_sizes, num_epochs, model_path,
          model_name, tensorboard):
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
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)

                    labels = labels.unsqueeze(1)
                    loss = criterion(outputs, labels.float())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()  # * inputs.size(0)
                running_corrects += torch.sum(torch.round(outputs).long() == labels.data).double()

                # del outputs, loss, labels
                # torch.cuda.empty_cache()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if tensorboard == 'on':
                writer.add_scalar("Loss/{}".format(phase), epoch_loss, epoch)
                writer.add_scalar("Acc/{}".format(phase), epoch_acc, epoch)

            if phase == 'val':
                lr_scheduler.step()  # (metrics=epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                model_filepath = str(folder / 'model_{model_name}_acc{best_acc:.4f}'.format(**vars()))
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

    # start writer for tensorboard (if 'on')
    if tensorboard == 'on':
        writer = SummaryWriter(log_path)

    # build dataset, load it onto dataloader, and get relevant information (dataset size, class names)
    is_local = torch.cuda.device_count() == 1 # todo: use data parallel
    loader_name = "base_loader_params"
    if not is_local:
        loader_name = "strong_loader_params"

    used_split_types = ['train', 'val']
    dataloaders = build_dataloaders(data_dir, loader_name, used_split_types, verbose)
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in used_split_types}

    # set device to be either GPU (if available) or CPU (if GPU not available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create a resnet18 instance and load it onto device
    model = MyResNet18().to(device)

    # # freeze model and only unfreeze last fc layers
    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # for name, param in model.named_parameters():
    #     if ('fc1' in name) or ('classifier.1' in name) or ('features.12' in name) or\
    #             ('features.11' in name) or ('features.10' in name) or ('features.9' in name) \
    #             or ('features.7' in name):
    #         param.requires_grad = True

    # specify loss function for model
    criterion = nn.CrossEntropyLoss().to(device)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # add weight decay here, if needed

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)

    # Train model
    model = train(model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, device, dataset_sizes, num_epochs=num_epochs, model_path=model_path, model_name=model_name, tensorboard=tensorboard)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str, default='/home/matan/models/my_models/chess',
                        help='location of model to be used')
    parser.add_argument('--model_name', type=str,
                        default='10_09_22_exp1',
                        help='name of model to be used')
    parser.add_argument('--data_dir', type=str, default='/home/matan/data/mydata/chess/caissabase/pgn/estat',
                        help='location of folder of images to be trained and validated')
    parser.add_argument('--tensorboard', type=str, default='off', help='start loss/acc tracking using tensorboard')
    parser.add_argument('--log_path', type=str, default='runs/chess/val_logs', help='folder of tensorboard logs')
    parser.add_argument('--learning_rate', type=int, default=0.0001, help='value of learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='step size of epochs for learning decay')
    parser.add_argument('--step_size', type=int, default=10, help='step size of epochs for learning decay')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor')
    parser.add_argument('--batch_size', type=int, default=20, help='recommended>=32 and not too big.')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    main()

