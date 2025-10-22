import numpy as np
from HOG_Model import Model
from pathlib import Path
from torchvision import transforms as T
from Utils.PCAMdataset import PCAMdataset
from Utils.Traning import run_experiment
from skimage.feature import hog
import json


with open(Path(__file__).parent.parent.parent.joinpath('./dataset/pcam/feature_mean_std.json'), 'r') as f:
    HOG_FEATURE_STATS = json.load(f)
    f_mean = HOG_FEATURE_STATS["HOG"]["mean"]
    f_std = HOG_FEATURE_STATS["HOG"]["std"]


####### Feature Extraction Function #######
def feature_transform(img:np.ndarray) -> np.ndarray:
    """ Example feature transformation: (C, H, W) """
    img = img.mean(axis=0)  # Convert to grayscale (H, W)
    # img = np.transpose(img, (1, 2, 0)) # Convert to (H, W, C) for skimage
    fd = hog(
            img.astype(int),
            orientations=12,
            pixels_per_cell=(24, 24),
            cells_per_block=(2, 2),
            visualize=False,
            channel_axis=None,
            )
    fd = (fd - f_mean) / f_std
    return fd



if __name__ == '__main__':

    ####### Hyperparameters and Data Loading #######
    N_RUNS = 5
    BATCH_SIZE = 512
    N_EPOCHS = 20

    mean = [0.7008, 0.5384, 0.6916]
    std = [0.2350, 0.2774, 0.2129]
    # Define transforms
    train_tf = T.Compose([
        #T.Grayscale(num_output_channels=1), # convert to grayscale
        T.Normalize(mean, std), # standardize
    ])

    eval_tf  = T.Compose([
        T.Normalize(mean, std),
    ])

    # Setting up directory
    path_dir = Path(__file__).parent.parent.parent.joinpath('./dataset/')
    print(f'Using data from: {path_dir}')

    # Create datasets
    train_data = PCAMdataset(
        x_path=path_dir / 'pcam' / 'camelyonpatch_level_2_split_train_x.h5',
        y_path=path_dir / 'pcam' /'camelyonpatch_level_2_split_train_y.h5',
        f_transform=feature_transform,
        transform=train_tf
    )

    test_data = PCAMdataset(
        x_path=path_dir / 'pcam' / 'camelyonpatch_level_2_split_test_x.h5',
        y_path=path_dir / 'pcam' / 'camelyonpatch_level_2_split_test_y.h5',
        f_transform=feature_transform,
        transform=eval_tf
    )


    run_experiment(Model, train_data, test_data, BATCH_SIZE, N_EPOCHS, N_RUNS)
