import numpy as np
from CNN_Model import Model
from pathlib import Path
from torchvision import transforms as T
from Utils.PCAMdataset import PCAMdataset
from Utils.Traning import run_experiment



if __name__ == '__main__':

    ####### Hyperparameters and Data Loading #######
    N_RUNS = 2
    BATCH_SIZE = 512
    N_EPOCHS = 1

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
        transform=train_tf
    )

    test_data = PCAMdataset(
        x_path=path_dir / 'pcam' / 'camelyonpatch_level_2_split_test_x.h5',
        y_path=path_dir / 'pcam' / 'camelyonpatch_level_2_split_test_y.h5',
        transform=eval_tf
    )


    run_experiment(Model, train_data, test_data, BATCH_SIZE, N_EPOCHS, N_RUNS)
