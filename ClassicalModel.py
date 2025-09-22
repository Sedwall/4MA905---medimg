import torch as pt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as knn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Load tensors
x_np = pt.load("/home/p3_medimg/Project/dataset/pcam_HOG/test_x.pt").numpy()
y_np = pt.load("/home/p3_medimg/Project/dataset/pcam_HOG/test_y.pt").numpy().reshape(-1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(x_np, y_np, test_size=0.2, random_state=42)

# Train a simple classifier
clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluate the classifier
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))