import numpy as np
from Pimg_Model import Model
from pathlib import Path
from torchvision import transforms as T
from Utils.PCAMdataset import PCAMdataset
from Utils.Traning import traning_run
from torch import nn, optim

# Import TDA pipeline requirements
from sklearn.pipeline import Pipeline
from gudhi.sklearn.cubical_persistence import CubicalPersistence
from gudhi.representations import PersistenceImage, DiagramSelector

# # Setting up directory
path_dir = Path(__file__).parent.parent.parent.joinpath('./dataset/')
# f_transform = get_feature_extractor(path_dir)

# --- Define this at the top level (so it can be pickled) ---
def feature_transform(data: np.ndarray) -> np.ndarray:
    gray_scale = data.mean(axis=0)  # Convert to grayscale
    feature_pipe = Pipeline([
        ("cub_pers", CubicalPersistence(homology_dimensions=0, n_jobs=None)),
        ("finite_diags", DiagramSelector(use=True, point_type="finite")),
        ("pers_img", PersistenceImage(
            bandwidth=50,
            weight=lambda x: x[1] ** 2,
            im_range=[0, 256, 0, 256],
            resolution=[5, 6],
        )),
    ])
    feature_vector = feature_pipe.fit_transform([gray_scale])
    return feature_vector[0]


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

        ## Define loss function and optimizer
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
