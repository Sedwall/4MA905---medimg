import numpy as np
from Stat_Model import Model
from pathlib import Path
from torchvision import transforms as T
from Utils.PCAMdataset import PCAMdataset
from Utils.Traning import traning_run
from sklearn.pipeline import Pipeline
from torch import nn, optim

# Import TDA pipeline requirements
from gudhi.sklearn.cubical_persistence import CubicalPersistence
from gudhi.representations import PersistenceImage, DiagramSelector, DimensionSelector

def stats(x):  
        if len(x) == 0:
            return np.zeros(9)
        return np.array([
            np.mean(x),
            np.std(x),
            np.median(x),
            np.percentile(x, 75) - np.percentile(x, 25),
            np.max(x) - np.min(x),
            np.percentile(x, 10),
            np.percentile(x, 25),
            np.percentile(x, 75),
            np.percentile(x, 90)
            ])

#### Persistence Statistics Function For Feature Vector #######
def persistence_statistics(diagram):
    """Compute persistence statistics as defined in your book."""
    
    p = diagram[:, 0]  # births
    q = diagram[:, 1]  # deaths
    mid = (p + q) / 2
    life = q - p

    # Compute all distributions
    features = stats(p)
    np.concat((features, stats(q)))
    np.concat((features, stats(mid)))
    np.concat((features, stats(life)))

    # Number of bars
    np.append(features, len(life))

    # Entropy

    L_mu = np.sum(life)
    if L_mu > 0 and life.size != 0:
        probs = life / L_mu
        np.append(features, -np.sum(probs * np.log(probs)))
    else:
        np.append(features, 0)

    return features


####### Feature Extraction Function #######
def feature_transform(img: np.ndarray) -> dict:
    """Transform an image into persistence statistics features."""
    img = img.mean(axis=0)  # Convert to grayscale by averaging channels
    # pipe_H0 = Pipeline([
    #     ("cub_pers", CubicalPersistence(
    #         homology_dimensions=0,
    #         homology_coeff_field=11,
    #         n_jobs=None
    #     )),
    #     ("finite_diags", DiagramSelector(use=True, point_type="finite")),
    # ])

    # diagrams_H0 = pipe_H0.fit_transform([img])
    # diagrams_H0 = np.array(diagrams_H0)[0]  # Extract the first (and only) diagram
    # features_H0 = persistence_statistics(diagrams_H0)

    pipe_H1 = Pipeline([
        ("cub_pers", CubicalPersistence(
            homology_dimensions=1,
            homology_coeff_field=11,
            n_jobs=None
        )),
        ("finite_diags", DiagramSelector(use=True, point_type="finite")),
    ])

    diagrams_H1 = pipe_H1.fit_transform([img])
    diagrams_H1 = np.array(diagrams_H1)[0]
    features_H1 = persistence_statistics(diagrams_H1)
    # return np.concatenate((features_H0, features_H1))
    return features_H1

if __name__ == '__main__':

    ####### Hyperparameters and Data Loading #######
    N_RUNS = 1
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
