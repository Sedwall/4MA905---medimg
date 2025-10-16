from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from Utils.PCAMdataset import get_entire_dataset
from Utils.Evaluate import Evaluate
from pathlib import Path
import numpy as np
from skimage.feature import hog
from time import time


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
    DATA_PATH = Path(__file__).parent.parent.parent.joinpath('./dataset/pcam')

    _, X_train, y_train= get_entire_dataset(x_path=DATA_PATH / "camelyonpatch_level_2_split_train_x.h5",
                                        y_path=DATA_PATH / "camelyonpatch_level_2_split_train_y.h5",
                                        feature_transform=feature_transform)
    print(f"Feature shape: {tuple(X_train.shape)}, Labels shape: {tuple(y_train.shape)}")

    # Train a simple classifier
    start = time()
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
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
    evaluator.save_metrics(metrics, Path("./HOG_LR_metrics.txt"))