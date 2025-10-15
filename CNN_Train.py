import numpy as np
from CNN_Model import Model
from pathlib import Path
from torchvision import transforms as T
from Utils.PCAMdataset import PCAMdataset
from Utils.Traning import traning_run




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
        transform=train_tf
    )

    test_data = PCAMdataset(
        x_path=path_dir / 'pcam' / 'camelyonpatch_level_2_split_test_x.h5',
        y_path=path_dir / 'pcam' / 'camelyonpatch_level_2_split_test_y.h5',
        transform=eval_tf
    )


    ####### Traning Of Model #######
    AVG_metrics = {}
    for i in range(N_RUNS):
        model = Model(chanels=16, dropout=0.5)
        model, metrics, evaluator = traning_run(model, train_data, test_data, BATCH_SIZE, N_EPOCHS)

        if not Path(__file__).parent.joinpath("runs").exists():
            Path(__file__).parent.joinpath("runs").mkdir()
        evaluator.save_metrics(metrics, Path(__file__).parent / "runs" / f"metrics{i}.txt")

        for key, value in zip(metrics.keys(), metrics.values()):
            if key in AVG_metrics.keys() and isinstance(value, float):
                AVG_metrics[key] += value
            else:
                AVG_metrics[key] = value


    for key, value in zip(AVG_metrics.keys(), AVG_metrics.values()):
        if isinstance(value, float):
            AVG_metrics[key] /= N_RUNS

    print(f"{'*' * 7:s}Final Metrics{'*' * 7:s}")
    evaluator.print_metrics(metrics)
    file_name = str(__file__).split('/')[-1].split('.')[0]
    evaluator.save_metrics(metrics, Path(__file__).parent/ f"{file_name}_final_metrics.txt")
