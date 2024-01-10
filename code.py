import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Lambda, Compose


DEBUG = True


class CustomImageDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def gen_csv(img_dir, path_csv):
    if img_dir[-1] == "/":
        img_dir.pop()

    img_nobody = os.listdir(img_dir+'/nobody')
    img_no_face = os.listdir(img_dir+'/no_face')
    img_people = os.listdir(img_dir+'/people')

    img_list_name = img_nobody + img_no_face + img_people
    list_labels = [0 for k in range(len(img_nobody))] + [1 for k in range(len(img_no_face))] + [2 for k in range(len(img_people))]

    dict_data = {"name": img_list_name, "label": list_labels}

    csv = pd.DataFrame(data=dict_data)
    if DEBUG:
        print(csv)

    csv.to_csv(index=False)