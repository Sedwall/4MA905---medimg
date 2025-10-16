from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from Utils.PCAMdataset import get_entire_dataset
from Utils.Evaluate import Evaluate
from pathlib import Path
import numpy as np
from time import time
# Import TDA pipeline requirements
from sklearn.pipeline import Pipeline
from gudhi.sklearn.cubical_persistence import CubicalPersistence
from gudhi.representations import PersistenceImage, DiagramSelector


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
            resolution=[10, 10],
        )),
    ])
    feature_vector = feature_pipe.fit_transform([gray_scale])
    return feature_vector[0]

if __name__ == '__main__':
    DATA_PATH = Path(__file__).parent.parent.parent.joinpath('./dataset/pcam')

    _, X_train, y_train= get_entire_dataset(x_path=DATA_PATH / "camelyonpatch_level_2_split_train_x.h5",
                                        y_path=DATA_PATH / "camelyonpatch_level_2_split_train_y.h5",
                                        feature_transform=feature_transform)
    print(f"Feature shape: {tuple(X_train.shape)}, Labels shape: {tuple(y_train.shape)}")

    # Train a simple classifier
    start = time()
    clf = LogisticRegression(max_iter=30_000, n_jobs=-1)
    clf.fit(X_train, y_train)
    elapsed = time() - start
    del X_train, y_train

    _, X_test, y_test = get_entire_dataset(x_path=DATA_PATH / "camelyonpatch_level_2_split_test_x.h5",
                                        y_path=DATA_PATH / "camelyonpatch_level_2_split_test_y.h5",
                                        feature_transform=feature_transform)
    y_pred = clf.predict(X_test)

    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    # For binary classification, use y_pred[:, 1] as probability for class 1
    roc_auc = roc_auc_score(y_test, y_pred)

    metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "training_time": elapsed
    }

    evaluator = Evaluate(model=None, val_data=None, device=None)
    evaluator.print_metrics(metrics)
    file_name = str(__file__).split('/')[-1].split('.')[0]
    evaluator.save_metrics(metrics,  Path(__file__).parent/ f"{file_name}_final_metrics.txt")