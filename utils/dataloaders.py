import os
import random
import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image
from utils import transforms as tr

def label_loader(label_path):
    label = cv2.imread(label_path, 0) / 255   # flag=0:  Return a grayscale image.
    return label

def full_path_loader(data_dir):
    train_data = [i for i in os.listdir(data_dir + 'train/A/') if not
    i.startswith('.')]
    train_data.sort()

    valid_data = [i for i in os.listdir(data_dir + 'val/A/') if not
    i.startswith('.')]
    valid_data.sort()

    train_label_paths = []
    val_label_paths = []
    for img in train_data:
        train_label_paths.append(data_dir + 'train/OUT/' + img)
    for img in valid_data:
        val_label_paths.append(data_dir + 'val/OUT/' + img)

    train_data_path = []
    val_data_path = []
    for img in train_data:
        train_data_path.append([data_dir + 'train/', img])
    for img in valid_data:
        val_data_path.append([data_dir + 'val/', img])

        
    train_dataset = {}
    val_dataset = {}
    for cp in range(len(train_data)):
        train_dataset[cp] = {'image': train_data_path[cp],
                         'label': train_label_paths[cp]}
    for cp in range(len(valid_data)):
        val_dataset[cp] = {'image': val_data_path[cp],
                         'label': val_label_paths[cp]}


    return train_dataset, val_dataset

def full_test_loader(data_dir):

    test_data = [i for i in os.listdir(data_dir + 'test/A/') if not
                    i.startswith('.')]
    test_data.sort()

    test_label_paths = []
    for img in test_data:
        test_label_paths.append(data_dir + 'test/OUT/' + img)

    test_data_path = []
    for img in test_data:
        test_data_path.append([data_dir + 'test/', img])

    test_dataset = {}
    for cp in range(len(test_data)):
        test_dataset[cp] = {'image': test_data_path[cp],
                           'label': test_label_paths[cp]}

    return test_dataset

def full_show_loader(data_dir):

    show_data = [i for i in os.listdir(data_dir + 'show/A/') if not
                    i.startswith('.')]
    show_data.sort()

    show_label_paths = []
    for img in show_data:
        show_label_paths.append(data_dir + 'show/OUT/' + img)

    show_data_path = []
    for img in show_data:
        show_data_path.append([data_dir + 'show/', img])

    show_dataset = {}
    for cp in range(len(show_data)):
        show_dataset[cp] = {'image': show_data_path[cp],
                           'label': show_label_paths[cp]}

    return show_dataset

def cdd_loader(img_path, label_path, aug):
    dir = img_path[0]
    name = img_path[1]

    img1 = Image.open(dir + 'A/' + name)
    img2 = Image.open(dir + 'B/' + name)
    label = Image.open(label_path)
    sample = {'image': (img1, img2), 'label': label}

    if aug:
        sample = tr.train_transforms(sample)
    else:
        sample = tr.test_transforms(sample)

    return sample['image'][0], sample['image'][1], sample['label']


class CDDloader(data.Dataset):

    def __init__(self, full_load, aug=False):
        random.shuffle(full_load)

        self.full_load = full_load
        self.loader = cdd_loader
        self.aug = aug

    def __getitem__(self, index):

        img_path, label_path = self.full_load[index]['image'], self.full_load[index]['label']

        return self.loader(img_path,
                           label_path,
                           self.aug)

    def __len__(self):
        return len(self.full_load)
