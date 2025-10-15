import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from torchvision import transforms as T
from tqdm import tqdm
from torch.utils.data import DataLoader

class PCAMdataset(Dataset):
    def __init__(self, x_path, y_path, f_transform=None, transform=None, target_transform=None):
        self.x_path = x_path
        self.y_path = y_path
        self.f_transform = f_transform
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

    
        # file handles for caching and lazy loading
        self._xf = self._xd = None  
        self._yf = self._yd = None  

    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        img, feature, label = 0, 0, 0
        self._ensure_open()
        # Load image and preprocess
        img = torch.from_numpy(self._xd[idx]).float()
        img = img.permute(2, 0, 1)  # Change to (C, H, W)

        if self.f_transform is not None:
            feature = self.f_transform(img.numpy())
        img = img / 255.0  # Rescale to [0, 1]

        # Handle image transformations
        if self.transform is not None:
            img = self.transform(img)

        if self.y_path is not None:
            label_np = np.asarray(self._yd[idx]).squeeze()
            label = torch.as_tensor(label_np, dtype=torch.float32)
            # Handle label transformation (probably unnecessary for binary labels) 
            if self.target_transform is not None:
                label = self.target_transform(label)
        
        return img, feature, label

    def _ensure_open(self):
        """Ensure HDF5 files are open only once per worker."""
        if self._xf is None:
            self._xf = h5py.File(self.x_path, "r")
            self._xd = self._xf["x"]
        if self.y_path and self._yf is None:
            self._yf = h5py.File(self.y_path, "r")
            self._yd = self._yf["y"]



def get_feature_dataset(x_path, y_path, feature_transform=None, transform=None) -> tuple[torch.Tensor, torch.Tensor]:
    # Create datasets
    train_data = PCAMdataset(
        x_path=x_path,
        y_path=y_path,
        transform=transform,
        f_transform=feature_transform,
    )
    
    dl = DataLoader(train_data, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)
    features = None
    labels = None
    for x, f, l in tqdm(dl, desc="Extracting features"):
        # Extend features and labels
        if features is None:
            features = f
            labels = l
        else:
            features = torch.cat((features, f), dim=0)
            labels = torch.cat((labels, l), dim=0)
    return x, features, labels