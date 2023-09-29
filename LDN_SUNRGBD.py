#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2023/9/29 12:35
# Author  : dongchao
# File    : LDN_SUNRGBD.py
# Software: PyCharm

import os

from imageio.v2 import imread
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from LDN_transforms import LDN_Normalize, LDN_ToTensor, LDN_ScaleNorm

img_dir_train_file = './LDN_data_SUNRGBD/img_dir_train.txt'
depth_dir_train_file = './LDN_data_SUNRGBD/depth_dir_train.txt'
label_dir_train_file = './LDN_data_SUNRGBD/label_train.txt'
img_dir_test_file = './LDN_data_SUNRGBD/img_dir_test.txt'
depth_dir_test_file = './LDN_data_SUNRGBD/depth_dir_test.txt'
label_dir_test_file = './LDN_data_SUNRGBD/label_test.txt'

import warnings

warnings.filterwarnings("ignore")


class SUNRGBD(Dataset):
    def __init__(self, transform=None, phase_train=True, data_dir=None):
        self.phase_train = phase_train
        self.transform = transform
        try:
            with open(img_dir_train_file, 'r') as f:
                self.img_dir_train = f.read().splitlines()
            with open(depth_dir_train_file, 'r') as f:
                self.depth_dir_train = f.read().splitlines()
            with open(label_dir_train_file, 'r') as f:
                self.label_dir_train = f.read().splitlines()

            with open(img_dir_test_file, 'r') as f:
                self.img_dir_test = f.read().splitlines()
            with open(depth_dir_test_file, 'r') as f:
                self.depth_dir_test = f.read().splitlines()
            with open(label_dir_test_file, 'r') as f:
                self.label_dir_test = f.read().splitlines()
        except:
            print("开始生成txt文件...")
            if data_dir is None:
                data_dir = r'E:\githubpro\dataset\sun_rgbd'

            self.img_dir_train = []
            self.depth_dir_train = []
            self.label_dir_train = []

            self.img_dir_test = []
            self.depth_dir_test = []
            self.label_dir_test = []

            depthpath = os.path.join(data_dir, os.listdir(data_dir)[0])
            imagepath = os.path.join(data_dir, os.listdir(data_dir)[1])
            labelpath = os.path.join(data_dir, os.listdir(data_dir)[-1])

            itempath_train = os.path.join(imagepath, "train")
            itempath_test = os.path.join(imagepath, "test")
            for item in os.listdir(itempath_train):
                self.img_dir_train.append(os.path.join(itempath_train, item))
            for item in os.listdir(itempath_test):
                self.img_dir_test.append(os.path.join(itempath_test, item))

            itempath_train = os.path.join(depthpath, "train")
            itempath_test = os.path.join(depthpath, "test")
            for item in os.listdir(itempath_train):
                self.depth_dir_train.append(os.path.join(itempath_train, item))
            for item in os.listdir(itempath_test):
                self.depth_dir_test.append(os.path.join(itempath_test, item))

            itempath_train = os.path.join(labelpath, "train")
            itempath_test = os.path.join(labelpath, "test")
            for item in os.listdir(itempath_train):
                self.label_dir_train.append(os.path.join(itempath_train, item))
            for item in os.listdir(itempath_test):
                self.label_dir_test.append(os.path.join(itempath_test, item))

            with open(img_dir_train_file, 'w') as f:
                f.write('\n'.join(self.img_dir_train))
            with open(depth_dir_train_file, 'w') as f:
                f.write('\n'.join(self.depth_dir_train))
            with open(label_dir_train_file, 'w') as f:
                f.write('\n'.join(self.label_dir_train))
            with open(img_dir_test_file, 'w') as f:
                f.write('\n'.join(self.img_dir_test))
            with open(depth_dir_test_file, 'w') as f:
                f.write('\n'.join(self.depth_dir_test))
            with open(label_dir_test_file, 'w') as f:
                f.write('\n'.join(self.label_dir_test))

    def __len__(self):
        if self.phase_train:
            return len(self.img_dir_train)
        else:
            return len(self.img_dir_test)

    def __getitem__(self, idx):
        if self.phase_train:
            img_dir = self.img_dir_train
            depth_dir = self.depth_dir_train
            label_dir = self.label_dir_train
        else:
            img_dir = self.img_dir_test
            depth_dir = self.depth_dir_test
            label_dir = self.label_dir_test

        image = imread(img_dir[idx])
        depth = imread(depth_dir[idx])
        label = imread(label_dir[idx])

        sample = {'image': image, 'depth': depth, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    # ************ test dataset ************
    LDN_data_dir = r"E:\githubpro\dataset\sun_rgbd"
    LDN_transform = Compose([
        LDN_ScaleNorm(),  # resize
        LDN_ToTensor(),  # ndarrays to Tensors
        LDN_Normalize()  # Normalize
    ])
    train_data = SUNRGBD(phase_train=True, data_dir=LDN_data_dir, transform=LDN_transform)
    print(train_data.__len__())  # 5285
    print(train_data[0]["image"].shape, train_data[0]["depth"].shape, train_data[0]["label"].shape)

    test_data = SUNRGBD(phase_train=False, data_dir=LDN_data_dir, transform=LDN_transform)
    print(test_data.__len__())  # 5050
    print(test_data[0]["image"].shape, test_data[0]["depth"].shape, test_data[0]["label"].shape)

    print(train_data.__len__() + test_data.__len__())  # 10335


