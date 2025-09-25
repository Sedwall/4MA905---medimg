print("Good Day, setting up environment...")
import torch as pt
from pathlib import Path
from sklearn.model_selection import train_test_split

# Standard scikit-learn imports
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Import TDA pipeline requirements
from gudhi.sklearn.cubical_persistence import CubicalPersistence
from gudhi.representations import PersistenceImage, DiagramSelector, DimensionSelector

# Import for evaluation
from Evaluate import Evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Paths
DIR = Path(__file__).parent.parent.parent.joinpath("dataset")
DATASET_FOLDER = Path(DIR.joinpath("./pcam_pt/"))

# Load dataset
X_train = pt.load(DATASET_FOLDER.joinpath("test_x.pt")).numpy()
y_train = pt.load(DATASET_FOLDER.joinpath("test_y.pt")).numpy().ravel()  # flatten labels

X_test = pt.load(DATASET_FOLDER.joinpath("valid_x.pt"))[:1000].numpy()
y_test = pt.load(DATASET_FOLDER.joinpath("valid_y.pt"))[:1000].numpy().ravel()  # flatten labels

pipe = Pipeline(
    [
        ("cub_pers", CubicalPersistence(homology_dimensions=0, homology_coeff_field= 11, n_jobs=-2)),
        ("finite_diags", DiagramSelector(use=True, point_type="finite")),
        (
            "pers_img",
            PersistenceImage(bandwidth=50, weight=lambda x: x[1] ** 2, im_range=[0, 256, 0, 256], resolution=[10, 10]),
        ),
        ("scaler", StandardScaler()),
        ("linear", LogisticRegression(max_iter=10000, C=10, penalty="l2", solver="liblinear")),
    ]
)

# Learn from the train subset
print("Fitting TDA pipeline...")
pipe.fit(X_train, y_train)

# Evaluate
print("Evaluating TDA pipeline...")
y_pred = pipe.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred)

metrics_ = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc
}

Evaluate.print_metrics(metrics_)
Evaluate.save_metrics(metrics_, "metrics.txt")