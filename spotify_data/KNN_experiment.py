# K-nearest neighbours classifier to be used as a baseline
# Matthew Dean and Caleb Bessit
# 15 October 2025

import os
import time
import numpy as np
from setup import VERSION
from sklearn.neighbors import KNeighborsClassifier
from utils import load_data, classification_metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score


NUM_CLASSES = 3

# Load training and testing data
if VERSION=="popularity":
    X_train, y_train = load_data("train", NUM_CLASSES)
    X_test, y_test   = load_data("test", NUM_CLASSES)
elif VERSION=="genre":
    X_train, y_train = load_data("train_genre", NUM_CLASSES)
    X_test, y_test   = load_data("test_genre", NUM_CLASSES)

print(f"Using {VERSION} as target.")


# Create the KNN classifier

# Can modify based on what you think would work best. I experimented with some of these values but others took too long
neighbours = [1,2,3,4,5,10,20,50,100,250,500,1000,5000,10000]

results = []

# For each variant of the target variable, run the experiment
for variant in ['popularity', 'genre']:
    if variant=="popularity":
        X_train, y_train = load_data("train", NUM_CLASSES)
        X_test, y_test   = load_data("test", NUM_CLASSES)
    elif variant=="genre":
        X_train, y_train = load_data("train_genre", NUM_CLASSES)
        X_test, y_test   = load_data("test_genre", NUM_CLASSES)

    variant_results = []
    for neighbour in neighbours:
        knn = KNeighborsClassifier(n_neighbors=neighbour)

        # Train the model (it basically stores the training data)
        start = time.time_ns()

        knn.fit(X_train, y_train)

        train_time = (time.time_ns()-start)*(10**-9)

        y_pred = knn.predict(X_test)

        metrics = classification_metrics(y_test, y_pred)

        # We add an array with values [accuracy, bal_acc, precision, recall, f1, train_time] associated with this number of neighbours
        # Will later plot the metric values as a function of the number of neighbours
        variant_results.append( list(metrics.values()) + [train_time] )

    results.append(variant_results)



# Can choose something convenient below
save_path = "/scratch/dnxmat002/"

os.makedirs(save_path, exist_ok=True)

# Expect results to have a shape like (num_variants, len(neighbours), num_metrics), i.e. (2, 14, 6) with the current values
np.save(  os.path.join(save_path, "knn_neighbour_results.npy"), np.array(results) )