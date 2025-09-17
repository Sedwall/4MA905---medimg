import torch as pt
import numpy as np
from pathlib import Path
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from skimage.color import rgb2gray

# Paths
DIR = Path(__file__).parent.joinpath("dataset")
DATASET_FOLDER = Path(DIR.joinpath("./pcam_pt/"))

# Load dataset
# x_data = pt.load(DATASET_FOLDER.joinpath("train_x.pt")).numpy()
# y_data = pt.load(DATASET_FOLDER.joinpath("train_y.pt")).numpy().ravel()  # flatten labels
x_data = pt.load(DATASET_FOLDER.joinpath("test_x.pt")).numpy()
y_data = pt.load(DATASET_FOLDER.joinpath("test_y.pt")).numpy().ravel()  # flatten labels
print("Data shape:", x_data.shape)
print("Labels shape:", y_data.shape)
print(y_data)
# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)

# Train SVM (RBF kernel by default)
clf = svm.SVC(kernel="rbf", C=1.0, gamma="scale")
print("Training SVM...")
clf.fit(X_train, y_train)


# Evaluate
print("Evaluating SVM...")
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))