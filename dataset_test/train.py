import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import h5py
import numpy as np
from data.PCAMdataset import PCAMdataset
from data.utils import compute_mean_std, compute_gray_mean_std

# calculate mean and std once
# mean, std = compute_mean_std('data/pcam/camelyonpatch_level_2_split_train_x.h5')
# mean_gray, std_gray = compute_gray_mean_std('data/pcam/camelyonpatch_level_2_split_train_x.h5')

mean_gray = 0.6043
std_gray  = 0.2530
mean = [0.7008, 0.5384, 0.6916]
std = [0.2350, 0.2774, 0.2129]

# Define transforms
train_tf = T.Compose([
    T.Grayscale(num_output_channels=1), # convert to grayscale
    T.Normalize(mean_gray, std_gray), # standardize
])

eval_tf  = T.Compose([
    T.Normalize(mean_gray, std_gray),
])

# Create datasets
train_data = PCAMdataset(
    x_path='data/pcam/camelyonpatch_level_2_split_train_x.h5',
    y_path='data/pcam/camelyonpatch_level_2_split_train_y.h5',
    transform=train_tf
)

test_data = PCAMdataset(
    x_path='data/pcam/camelyonpatch_level_2_split_test_x.h5',
    y_path='data/pcam/camelyonpatch_level_2_split_test_y.h5',
    transform=eval_tf
)
