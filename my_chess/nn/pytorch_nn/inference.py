import time
import numpy as np
import torch
import argparse
import re
from copy import deepcopy
import pickle
import os

from nn.pytorch_nn.data_loading.dataloaders import build_dataloaders
from load_model import load_model
from utils.shopic_utils import jetson_tensorrt as jetson

BATCH_SIZE = 7


def add_to_set(label_sets, values, count):
    for v, c in zip(values.tolist(), count.tolist()):
        label_sets[v] = label_sets.get(v, 0) + c


def predict(dataloaders, filenames, model, device):
    total = 0
    frame_vote = {}
    frame_labels = {}
    for inputs, labels in dataloaders:
        if jetson:
            inputs = inputs.half()
        orig_inputs = inputs
        inputs_ = deepcopy(inputs)
        inputs_ = torch.flip(inputs_, [3])
        for inputs, labels in [(orig_inputs, labels), (inputs_, labels)]:
            print(inputs[0].shape)
            inputs = inputs.to(device)
            output = model(inputs)
            output = output.to(device)
            output = output.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            output = output.flatten()
            for index in range(output.shape[0]):
                m = re.search("(.*)\\\\(.*)\\\\frame([0-9]*)\.jpg", filenames[total + index])
                video_name = m.group(1)
                frame_number = eval(m.group(3))
                frame_vote[video_name] = frame_vote.get(video_name, {})
                frame_vote[video_name][frame_number] = frame_vote[video_name].get(frame_number, [])
                frame_vote[video_name][frame_number].append(output[index])
                frame_labels[video_name] = frame_labels.get(video_name, {})
                frame_labels[video_name][frame_number] = labels[index]
        total += output.shape[0]
    return frame_vote, frame_labels


def get_paths(intermediate_dir):
    path_frame_labels = os.path.join(intermediate_dir, 'frame_labels.pkl')
    path_frame_vote = os.path.join(intermediate_dir, 'frame_vote.pkl')
    return path_frame_labels, path_frame_vote


def get_predictions(dataloaders, filenames, model, device, intermediate_dir):
    path_frame_labels = None
    path_frame_vote = None
    if intermediate_dir:
        path_frame_labels, path_frame_vote = get_paths(intermediate_dir)
        os.makedirs(intermediate_dir, exist_ok=True)
    if intermediate_dir and os.path.exists(path_frame_labels):
        with open(path_frame_labels, "rb") as fp:
            frame_labels = pickle.load(fp)
        with open(path_frame_vote, "rb") as fp:
            frame_vote = pickle.load(fp)
    else:
        frame_vote, frame_labels = predict(dataloaders, filenames, model, device)
        save_to_file(intermediate_dir, frame_vote, frame_labels)
    return frame_vote, frame_labels


def save_to_file(intermediate_dir, frame_vote, frame_labels):
    path_frame_labels, path_frame_vote = get_paths(intermediate_dir)
    with open(path_frame_labels, "wb") as fp:
        pickle.dump(frame_labels, fp)
    with open(path_frame_vote, "wb") as fp:
        pickle.dump(frame_vote, fp)


def main(model_path, data_dir, intermediate_dir):
    correct = 0
    total = 0
    incorrect_label_sets = {}
    label_sets = {}
    start_time = time.time()
    if torch.cuda.device_count() >= 0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = load_model(model_path, device)
    image_datasets = build_dataloaders(data_dir, split_subset=['test'])
    dataloaders = torch.utils.data.DataLoader(image_datasets['test'], batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=4)

    filenames = []
    for img, _ in dataloaders.dataset.imgs:
        filenames.append(img.split('/')[-1])

    frame_vote, frame_labels = get_predictions(dataloaders, filenames, model, device, intermediate_dir)

    FRAMES_PER_PREDICTION = 1
    for k, v in frame_vote.items():
        frame_number = 0
        frame_labels_per_video = frame_labels[k]
        while frame_number in v:
            all_exist = True
            values = []
            labels = []
            for i in range(FRAMES_PER_PREDICTION):
                fn = frame_number + i
                if fn not in v:
                    all_exist = False
                    break
                values += v[fn]
                labels += [frame_labels_per_video[fn]]
            if not all_exist:
                frame_number += FRAMES_PER_PREDICTION
                continue

            values = np.array(values)
            gt_label = 1 if 1 in labels else 0
            if (np.array(labels) != labels[0]).any():
                frame_number += FRAMES_PER_PREDICTION
                continue
            # 0.75 is the percentage required for open vote
            predict_label = 1 if ((values >= 0.645).sum() >= values.size * 0.75) else 0
            total += 1
            label_sets[gt_label] = label_sets.get(gt_label, 0) + 1
            if predict_label == gt_label:
                correct += 1
            else:
                incorrect_label_sets[gt_label] = incorrect_label_sets.get(gt_label, 0) + 1
                print("incorrect prediction on video: {}, frames #: {}-{}, gt label: {}".format(
                    k, frame_number, frame_number + FRAMES_PER_PREDICTION-1, gt_label))
            frame_number += FRAMES_PER_PREDICTION

    print("acc: {}".format(float(correct) / total))
    print("total files: {}".format(len(filenames)))
    print('input data by labels:')
    for k, v in label_sets.items():
        print("{}: {}".format(dataloaders.dataset.classes[k], v))
    print('prediction mistakes by gt labels:')
    for k, v in incorrect_label_sets.items():
        print("{}: {}".format(dataloaders.dataset.classes[k], v))

    print("total time: {} seconds".format(time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/home/matan/models/child_seat/06_09_22_exp2/model_06_09_22_exp2_acc0.9996')
    # parser.add_argument('--model_path', default='/home/matan/models/child_seat/25_08_22_exp2/model_25_08_22_exp2_acc0.9969')
    parser.add_argument('--data_dir', default='/home/matan/data/child_seat/splitted')
    parser.add_argument('--intermediate_dir', default='')
    args = parser.parse_args()
    main(args.model_path, args.data_dir, args.intermediate_dir)
