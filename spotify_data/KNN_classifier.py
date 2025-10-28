# K-nearest neighbours classifier to be used as a baseline 
# Matthew Dean and Caleb Bessit
# 15 October 2025

import os
import time
import numpy as np
from setup import VERSION
from sklearn.neighbors import KNeighborsClassifier
from utils import load_data, classification_metrics

# Set path to save results to
save_path = "results/KNN/"
os.makedirs(save_path, exist_ok=True)

# Array for exploring the impact that the number of neighbours has on performance
neighbours = [1,2,3,4,5,10,20,50,100,250,500,1000,5000]

if VERSION=="popularity":
    num_class_list = [3,5]
elif VERSION=="genre":
    num_class_list = [5]

# Run the experiment different numbers of target classes and k values 
for num_class in num_class_list:
    if VERSION=="popularity":
        X_train, y_train = load_data("train", num_class)
        X_test, y_test   = load_data("test", num_class)
    elif VERSION=="genre":
        X_train, y_train = load_data("train_genre", num_class)
        X_test, y_test   = load_data("test_genre", num_class)

    results = []
    for neighbour in neighbours:
        knn = KNeighborsClassifier(n_neighbors=neighbour)

        # Train classifier and obtain training time
        start = time.time_ns()
        knn.fit(X_train, y_train)
        train_time = (time.time_ns()-start)*(10**-9)
        
        # test model
        y_pred = knn.predict(X_test)

        # obtain performance metrics
        metrics = classification_metrics(y_test, y_pred)
        
        # We add an array with values [accuracy, bal_acc, precision, recall, f1, train_time] associated with this number of neighbours
        results.append( list(metrics.values()) + [train_time] )

    # Expect results to have a shape like (num_variants, len(neighbours), num_metrics), i.e. (2, 14, 6) with the current values
    np.save(os.path.join(save_path, f"knn_{VERSION}_{num_class}.npy"), np.array(results) )