print("Good Day, setting up environment...")
import torch as pt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Standard scikit-learn imports
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Import TDA pipeline requirements
from gudhi.sklearn.cubical_persistence import CubicalPersistence
from gudhi.representations import PersistenceImage, DiagramSelector, DimensionSelector

# Paths
DIR = Path(__file__).parent.parent.parent.joinpath("dataset")
DATASET_FOLDER = Path(DIR.joinpath("./pcam_pt_TDA/"))

# Load dataset
x_data = pt.load(DATASET_FOLDER.joinpath("test_x.pt")).numpy()
y_data = pt.load(DATASET_FOLDER.joinpath("test_y.pt")).numpy().ravel()  # flatten labels

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)
pipe = Pipeline(
    [
        ("cub_pers", CubicalPersistence(homology_dimensions=0, homology_coeff_field= 11, n_jobs=-2)),
        # Or for multiple persistence dimension computation
        # ("cub_pers", CubicalPersistence(homology_dimensions=[0, 1])),
        # ("H0_diags", DimensionSelector(index=0)), # where index is the index in homology_dimensions array
        ("finite_diags", DiagramSelector(use=True, point_type="finite")),
        (
            "pers_img",
            PersistenceImage(bandwidth=50, weight=lambda x: x[1] ** 2, im_range=[0, 256, 0, 256], resolution=[10, 10]),
        ),
        ("scaler", StandardScaler()),
        ("linear", LogisticRegression(max_iter=10000, C=10, penalty="l2", solver="liblinear")),
    ]
)
# 

print("Fitting TDA pipeline...")
# Learn from the train subset
pipe.fit(X_train, y_train)
# Predict from the test subset
print("Predicting...")
y_pred = pipe.predict(X_test)

# Evaluate
print("Evaluating Logistic Regression...")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(
    "\nClassification Report:\n",
    classification_report(y_test, y_pred, zero_division=0)
)