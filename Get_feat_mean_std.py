from Utils.PCAMdataset import get_entire_dataset
from pathlib import Path
import numpy as np
from skimage.feature import hog
import json
import torch


# Import TDA pipeline requirements
from sklearn.pipeline import Pipeline
from gudhi.sklearn.cubical_persistence import CubicalPersistence
from gudhi.representations import PersistenceImage, DiagramSelector


####### Feature Extraction Function #######
def HOG_feature_transform(img:np.ndarray) -> np.ndarray:
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


# --- Define this at the top level (so it can be pickled) ---
def TDA_img_feature_transform(data: np.ndarray) -> np.ndarray:
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
    return feature_vector[0]


def get_mean_and_std(data: torch.Tensor) -> tuple[float, float]:
    feat_mean = torch.mean(data)
    feat_std = torch.std(data)
    print(feat_mean, feat_std)
    feat_mean, feat_std = feat_mean.item(), feat_std.item()
    print(feat_mean, feat_std)
    return feat_mean, feat_std



if __name__ == '__main__':
    DATA_PATH = Path(__file__).parent.joinpath('./dataset/pcam')

    _, X_train, _= get_entire_dataset(x_path=DATA_PATH / "camelyonpatch_level_2_split_train_x.h5",
                                        y_path=DATA_PATH / "camelyonpatch_level_2_split_train_y.h5",
                                        feature_transform=HOG_feature_transform)
    print(f"Feature shape: {tuple(X_train.shape)}")

    HOG_FEATURE_STATS = {}
    # Get feature mean and std
    feat_mean, feat_std = get_mean_and_std(X_train)
    HOG_FEATURE_STATS["HOG"] = {
        "mean": feat_mean,
        "std": feat_std
    }

    _, X_train, _= get_entire_dataset(x_path=DATA_PATH / "camelyonpatch_level_2_split_train_x.h5",
                                        y_path=DATA_PATH / "camelyonpatch_level_2_split_train_y.h5",
                                        feature_transform=TDA_img_feature_transform)


    # Get feature mean and std
    feat_mean, feat_std = get_mean_and_std(X_train)
    HOG_FEATURE_STATS["TDA_img"] = {
        "mean": feat_mean,
        "std": feat_std
    }

    # Save json file
    with open(DATA_PATH / "feature_mean_std.json", "w") as f:
        json.dump(HOG_FEATURE_STATS, f, indent=4)