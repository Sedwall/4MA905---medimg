import numpy as np
from Landscape_Model import Model
from pathlib import Path
from torchvision import transforms as T
from Utils.PCAMdataset import PCAMdataset
from Utils.Traning import run_experiment

# Import TDA pipeline requirements
from gudhi.sklearn.cubical_persistence import CubicalPersistence
from gudhi.representations import DiagramSelector, Landscape

# # Setting up directory
path_dir = Path(__file__).parent.parent.parent.joinpath('./dataset/')
# f_transform = get_feature_extractor(path_dir)

# --- Define this at the top level (so it can be pickled) ---
def feature_transform(data: np.ndarray) -> np.ndarray:
    gray_scale = data.mean(axis=0)  # Convert to grayscale

    num_landscapes = 5
    resolution = 100

    # Compute persistence diagram first
    cub_pers = CubicalPersistence(homology_dimensions=(0,1), n_jobs=None)
    diagrams = cub_pers.fit_transform([gray_scale])[0]

    selector = DiagramSelector(use=True, point_type="finite")
    landscape = Landscape(num_landscapes=num_landscapes, resolution=resolution)
    
    feature_parts = []
    for diag in diagrams:  # For H0 and H1
        # Check if diagram is empty or has no finite points
        if diag is None or len(diag) == 0 or not np.isfinite(diag).all():
            # Return zero vector of the expected landscape shape
            feature_parts.append(np.zeros(num_landscapes * resolution))
            continue
        
        finite_d = selector.fit_transform([diag])[0]
        if len(finite_d) == 0:
            feature_parts.append(np.zeros(num_landscapes * resolution))
            continue

        # Compute landscape
        vec = landscape.fit_transform([finite_d])[0]
        feature_parts.append(vec)

    # Concatenate H0 and H1 features
    feature_vector = np.concatenate(feature_parts)
    return feature_vector


if __name__ == '__main__':
    ####### Hyperparameters and Data Loading #######
    N_RUNS = 1
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


    run_experiment(Model, train_data, test_data, BATCH_SIZE, N_EPOCHS, N_RUNS)
