import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import h5py
import numpy as np

class PCAMdataset(Dataset):
    def __init__(self, x_path, y_path, feature_path=None, transform=None, target_transform=None):
        self.x_path = x_path
        self.y_path = y_path
        self.feature_path = feature_path
        self.transform = transform
        self.target_transform = target_transform

        # Determine dataset size and validate
        with h5py.File(self.x_path, "r") as fx:
            assert "x" in fx, "Expected key 'x' in HDF5"
            self.N = fx["x"].shape[0]

        if self.y_path:
            with h5py.File(self.y_path, "r") as fy:
                assert "y" in fy, "Expected key 'y' in HDF5"
                assert fy["y"].shape[0] == self.N, "x/y length mismatch"

        if self.feature_path != None:
            with h5py.File(self.feature_path, "r") as fy:
                assert "data" in fy, "Expected key 'data' in HDF5"
                assert fy["data"].shape[0] == self.N, "length mismatch"
        
        # file handles for caching and lazy loading
        self._xf = self._xd = None  
        self._yf = self._yd = None  
        self._ff = self._fd = None

    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        img, feature, label = 0, 0, 0
        self._ensure_open()
        # Load image and preprocess
        img = torch.from_numpy(self._xd[idx]).float()
        img = img / 255.0  # Rescale to [0, 1]
        img = img.permute(2, 0, 1)  # Change to (C, H, W)
        # Handle image transformations
        if self.transform is not None:
            img = self.transform(img)

        if self.y_path is not None:
            label_np = np.asarray(self._yd[idx]).squeeze()
            label = torch.as_tensor(label_np, dtype=torch.float32)
            # Handle label transformation (probably unnecessary for binary labels) 
            if self.target_transform is not None:
                label = self.target_transform(label)

        if self.feature_path is not None:
            feature = torch.from_numpy(self._fd[idx]).float()
        
        return img, feature, label

    def _ensure_open(self):
        """Ensure HDF5 files are open only once per worker."""
        if self._xf is None:
            self._xf = h5py.File(self.x_path, "r")
            self._xd = self._xf["x"]
        if self.y_path and self._yf is None:
            self._yf = h5py.File(self.y_path, "r")
            self._yd = self._yf["y"]
        if self.feature_path and self._ff is None:
            self._ff = h5py.File(self.feature_path, "r")
            self._fd = self._ff["data"]