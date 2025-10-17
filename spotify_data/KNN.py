# K-nearest neighbours classifier to be used as a baseline
# Matthew Dean
# 15 October 2025

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from utils import load_data

NUM_CLASSES = 3

# Load training and testing data
X_train, y_train = load_data("train", NUM_CLASSES)
X_test, y_test   = load_data("test", NUM_CLASSES)


# Create the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model (it basically stores the training data)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# Compute balanced accuracy
bal_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Accuracy: {bal_acc:.3f}")