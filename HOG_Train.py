import numpy as np
from HOG_Model import Model
from pathlib import Path
from torchvision import transforms as T
from Utils.PCAMdataset import PCAMdataset
from Utils.Traning import traning_run, metrics_avg
from skimage.feature import hog
from torch import nn, optim



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
    return fd



if __name__ == '__main__':

    ####### Hyperparameters and Data Loading #######
    N_RUNS = 10
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


    ####### Traning Of Model #######
    AVG_metrics = {}
    for i in range(N_RUNS):
        model = Model(chanels=16, dropout=0.5)
        ## Define loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        model, metrics, evaluator = traning_run(model, train_data, test_data, loss_fn, optimizer, BATCH_SIZE, N_EPOCHS)

        if not Path(__file__).parent.joinpath("runs").exists():
            Path(__file__).parent.joinpath("runs").mkdir()
        evaluator.save_metrics(metrics, Path(__file__).parent / "runs" / f"metrics{i}.txt")
        
        for key, value in zip(metrics.keys(), metrics.values()):
            if key in AVG_metrics.keys():
                AVG_metrics[key].append(value)
            else:
                AVG_metrics[key] = [value]


    # Calculate and print average metrics
    metrics_avg(evaluator, AVG_metrics, __file__)
