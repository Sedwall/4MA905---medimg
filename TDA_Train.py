import numpy as np
from TDA_Model import Model
from pathlib import Path
from torchvision import transforms as T
from Utils.PCAMdataset import PCAMdataset
from Utils.Training import run_experiment
import json

# Import TDA pipeline requirements
from sklearn.pipeline import Pipeline
from gudhi.sklearn.cubical_persistence import CubicalPersistence
from gudhi.representations import PersistenceImage, DiagramSelector

# Setting up directory
path_dir = Path(__file__).parent.joinpath('./dataset/')


with open(path_dir.joinpath('./pcam/feature_mean_std.json'), 'r') as f:
    HOG_FEATURE_STATS = json.load(f)
    f_mean = HOG_FEATURE_STATS["TDA_img"]["mean"]
    f_std = HOG_FEATURE_STATS["TDA_img"]["std"]



# --- Define this at the top level (so it can be pickled) ---
def feature_transform(data: np.ndarray) -> np.ndarray:
    gray_scale = data.mean(axis=0)  # Convert to grayscale

    feature_pipe = Pipeline([
        ("cub_pers", CubicalPersistence(homology_dimensions=0, n_jobs=None)),
        ("finite_diags", DiagramSelector(use=True, point_type="finite")),
        ("pers_img", PersistenceImage(
                bandwidth=25,
                weight=lambda x: x[1],
                im_range=[0, 256, 0, 256],
                resolution=[16, 16],
                        )),
        ])

    feature_vector = feature_pipe.fit_transform([gray_scale])
    feature_vector = (feature_vector[0] - f_mean) / f_std
    return feature_vector


if __name__ == '__main__':
    ####### Hyperparameters and Data Loading #######
    N_RUNS = 5
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

    run_experiment(Model, train_data, test_data, BATCH_SIZE, N_EPOCHS, N_RUNS, __file__)
