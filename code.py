import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import utils.file

from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Lambda, Compose


DEBUG = True


class CustomImageDataset(Dataset):
    def __init__(self, csv_dir, img_dir, transform=None, target_transform=None):
        utils.file.gen_csv(img_dir, csv_dir)
        if csv_dir[-1] == '/':
            csv_dir.pop()
        annotation_file = csv_dir + "/labels.csv"

        self.img_labels = pd.read_csv(annotation_file)
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
