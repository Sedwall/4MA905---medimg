import numpy as np
from ATOL_Model import Model
from pathlib import Path
from torchvision import transforms as T
from PIL import Image
from Utils.PCAMdataset import PCAMdataset, get_entire_dataset
from Utils.Traning import traning_run
from tqdm import tqdm
from sklearn.cluster import KMeans
from gudhi.representations.vector_methods import Atol
from gudhi import CubicalComplex
import pickle
import torch
from torch import nn, optim


# # Setting up directory
path_dir = Path(__file__).parent.parent.parent.joinpath('./dataset/')
# f_transform = get_feature_extractor(path_dir)

with open(Path(__file__).parent / "atol_vectoriser.pkl", "rb") as file:
        atol_vectoriser = pickle.load(file)

def image_to_persistence(img: np.ndarray,
                         homology_coeff: int = 11,
                         min_persistence: float = 0.0) -> np.ndarray:
    """
    Convert 2D grayscale image `img` (shape [H, W]) to a persistence diagram.
    Returns an (N, 2) array of (birth, death) pairs (for all homology dimensions).
    """
    # Optionally invert the intensities so that bright features become low
    H, W = img.shape
    # Flatten top-dimensional cells
    flat = img.flatten()
    # Build cubical complex
    cc = CubicalComplex(dimensions=[H, W], top_dimensional_cells=flat)
    # Compute persistence
    cc.compute_persistence(homology_coeff_field=homology_coeff,
                           min_persistence=min_persistence)
    # Get diagram
    diag = cc.persistence()
    diag = [np.array(pair[1]) for pair in diag if pair[0] == 0]  # Keep only H0
    return np.array(diag)

# --- Define this at the top level (so it can be pickled) ---
def feature_transform(data: torch.Tensor, atol_vectoriser=atol_vectoriser) -> np.ndarray:
    
    diag = image_to_persistence(data[0], homology_coeff=11, min_persistence=0.0)
    feature_vector = atol_vectoriser.transform([diag])
    return feature_vector.squeeze(0)


if __name__ == '__main__':
    ####### Hyperparameters and Data Loading #######
    N_RUNS = 10
    BATCH_SIZE = 512
    N_EPOCHS = 20


    mean = [0.7008, 0.5384, 0.6916]
    std = [0.2350, 0.2774, 0.2129]
    # Define transforms
    train_tf = T.Compose([
        T.Normalize(mean, std), # standardize
        # T.Grayscale(num_output_channels=1), # convert to grayscale
    ])

    eval_tf  = T.Compose([
        T.Normalize(mean, std),
        # T.Grayscale(num_output_channels=1), # convert to grayscale
    ])

    
    print(f'Using data from: {path_dir}')

    # Create datasets
    train_data = PCAMdataset(
        x_path=path_dir / 'pcam' / 'camelyonpatch_level_2_split_train_x.h5',
        y_path=path_dir / 'pcam' /'camelyonpatch_level_2_split_train_y.h5',
        transform=train_tf,
        f_transform= feature_transform
    )

    test_data = PCAMdataset(
        x_path=path_dir / 'pcam' / 'camelyonpatch_level_2_split_test_x.h5',
        y_path=path_dir / 'pcam' / 'camelyonpatch_level_2_split_test_y.h5',
        transform=eval_tf,
        f_transform= feature_transform
    )


    ####### Traning Of Model #######
    AVG_metrics = {}
    for i in range(N_RUNS):
        model = Model(chanels=16, dropout=0.5)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        model, metrics, evaluator = traning_run(model, train_data, test_data, loss_fn, optimizer, BATCH_SIZE, N_EPOCHS)

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
    evaluator.print_metrics(AVG_metrics)
    file_name = str(__file__).split('/')[-1].split('.')[0]
    evaluator.save_metrics(AVG_metrics, Path(__file__).parent/ f"{file_name}_final_metrics.txt")
