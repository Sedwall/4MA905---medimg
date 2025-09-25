import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import h5py
import numpy as np
from data.PCAMdataset import PCAMdataset

def compute_mean_std(train_x_path, pre_transform=None, batch_size=2048, num_workers=4):
    # pre_transform can include grayscale, flips/crops are okay too (they don't change channel count)
    ds = PCAMdataset(train_x_path, y_path=None, transform=pre_transform)  # returns [C,H,W] in [0,1]
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True, persistent_workers=True)

    s  = None   # sum over all pixels
    s2 = None   # sum of squares
    n_pixels = 0

    for x in dl:                       # x: [B,C,H,W]
        x = x.to(torch.float64)
        if s is None:
            C = x.size(1)
            s  = torch.zeros(C, dtype=torch.float64)
            s2 = torch.zeros(C, dtype=torch.float64)
        b, c, h, w = x.shape
        s  += x.sum(dim=(0,2,3))
        s2 += (x**2).sum(dim=(0,2,3))
        n_pixels += b*h*w

    mean = (s / n_pixels).to(torch.float32)                    # (C,)
    var  = (s2 / n_pixels) - (mean.to(torch.float64)**2)
    std  = torch.sqrt(var.clamp_min(1e-12)).to(torch.float32).clamp_min(1e-8)  # (C,)
    return mean.tolist(), std.tolist()

def compute_gray_mean_std(train_x_path, out_channels=1, bs=2048, nw=4):
    gray = T.Grayscale(num_output_channels=out_channels)
    ds = PCAMdataset(train_x_path, y_path=None, transform=gray)  # returns [C,H,W] in [0,1]
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    s = s2 = None; n = 0
    for x in dl:  # [B,C,H,W]
        x = x.double()
        if s is None:
            C = x.size(1)
            s  = torch.zeros(C, dtype=torch.float64)
            s2 = torch.zeros(C, dtype=torch.float64)
        b, c, h, w = x.shape
        s  += x.sum((0,2,3))
        s2 += (x**2).sum((0,2,3))
        n  += b*h*w
    mean = (s/n).float().tolist()
    std  = torch.sqrt((s2/n) - torch.tensor(mean, dtype=torch.float64)**2).float().clamp_min(1e-8).tolist()
    return mean, std