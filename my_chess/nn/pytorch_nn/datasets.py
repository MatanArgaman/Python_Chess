from __future__ import print_function, division
from torchvision import datasets, transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2


class AlbumentationsWrapper:
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, *args, **kwargs):
        x = args[0]
        x = self.aug(image=x)
        return x['image']


def get_transforms():
    data_transforms = {

        'train': transforms.Compose([

            transforms.ToTensor(),
            transforms.Normalize([0.4392, 0.3950, 0.3502], [0.2539, 0.2431, 0.2327]),
            transforms.RandomHorizontalFlip(0.5)
        ]),

        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4392, 0.3950, 0.3502], [0.2539, 0.2431, 0.2327])
        ]),
    }

    data_transforms['test'] = data_transforms['val']
    return data_transforms


def is_valid_file(f):
    return (f.endswith('jpg') or f.endswith('png')) and os.stat(f).st_size > 0


def build_datasets(data_dir, split_subset=['train', 'val', 'test'], verbose=False):
    data_transforms = get_transforms()
    if verbose:
        print('creating folders for empty classes, if they appear in any of the train or val or test')
    data_folders = []
    for b in split_subset:
        data_folders += glob.glob(data_dir + '/{}/*'.format(b))
    for folder in data_folders:
        for b in split_subset:
            os.makedirs(data_dir + '/{}/'.format(b) + folder.split('/')[-1], exist_ok=True)

    image_datasets = {}
    for split_type in split_subset:
        path = os.path.join(data_dir, split_type)
        print('split:', split_type, 'path:', path)
        transform = data_transforms[split_type]
        image_datasets[split_type] = datasets.ImageFolder(path, transform, loader=loader, is_valid_file=is_valid_file)
        if verbose:
            print('dataset of', split_type, 'len', len(image_datasets[split_type]), '#classes', len(image_datasets[split_type].classes), 'used data_dir is', data_dir)
    return image_datasets


def tensor_to_rgb(inp):
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def imshow(inp, title=None, figsize=(10, 10)):
    """Imshow for Tensor."""

    plt.figure(figsize=figsize)
    plt.imshow(tensor_to_rgb(inp))
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()


def folder_stats(data_dir, traintest):
    sizes = []
    for product_folder in glob.glob(data_dir + '/' + traintest + '/*'):
        images = glob.glob(product_folder + '/*.jpg')
        sizes.append(len(images))
        # print(product_folder.split('/')[-1], len(images))
    print(max(sizes), min(sizes), 'non-empty', sum([1 for size in sizes if size > 0]))


if __name__ == '__main__':
    # folder_stats('/Users/assaflehr/datasets/trolly_crops','val')
    ds = build_datasets('/home/matan/rep/flip_camera_detector/flip_camera_dataset', override_transforms_type='test_aug')['val']
    print(len(ds))
    for id in range(0, 3):  # range(0,len(ds),len(ds)//50):
        x, y = ds[id]
        imshow(x)
