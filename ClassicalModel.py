import torch as pt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as knn
import numpy as np
import h5py
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score


def load_data(file_path):
    # Load tensors
    with h5py.File(file_path, 'r') as f:
                key = str(list(f.keys())[-1])
                data = pt.tensor(np.array(f[key]), dtype=pt.float32)
    return data


X_train = load_data("/home/p3_medimg/Project/dataset/pcam_HOG_h5/train_x.h5").numpy()
y_train = load_data("/home/p3_medimg/Project/dataset/pcam/camelyonpatch_level_2_split_train_y.h5").squeeze().numpy()

# Train a simple classifier
clf = svm.SVC()
clf.fit(X_train, y_train)
del X_train, y_train

X_test = load_data("/home/p3_medimg/Project/dataset/pcam_HOG_h5/test_x.h5").numpy()
y_test = load_data("/home/p3_medimg/Project/dataset/pcam/camelyonpatch_level_2_split_test_y.h5").numpy()
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
"roc_auc": roc_auc
}

