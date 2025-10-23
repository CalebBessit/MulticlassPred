# Utility script to load the data
# Caleb Bessit
# 14 October 2025

import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from collections import Counter



def split_into_classes(data,num_classes):
    size = 100//num_classes

    # print(f"Class counts")
    print(f"Partioning data into {num_classes} classes.")
    for i in range(num_classes):
        lower, upper = size*i, size*(i+1)
        if i == (num_classes-1):
            upper = 100

        mask = (data>lower) & (data<=upper)

        print(f"\t+ For {i}: range is ({lower}, {upper}], count is {np.sum(mask)}")

        data[mask] = i

    print(f"\t Class counts: {dict(Counter(data))}")
    return data


#Calculate the sample weights for each point in a class
def calculate_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced",classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    sample_weight = np.array([class_weights[label] for label in y_train])
    return class_weights, sample_weight


def load_data(fileset, num_classes, features, target):
    print(f"Loading {fileset} data...")

    X, y = None, None

    data = pd.read_csv(f"{fileset}.csv")
    data = data.dropna()

    X = data[features].to_numpy()
    y = data[target].to_numpy(dtype=np.int32)[:,0]

    if target=="popularity":
        y = split_into_classes(y, num_classes)

    # Do masking to create values with the number of classes

    print("Done.\n")
    return X, y

def classification_metrics(y_true, y_pred, average="weighted"):
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, average=average, zero_division=0)
    

    print(f"\n  $$ Metrics $$")
    for metric, value in metrics.items():
        print(f"{metric:^20}: {value}")
    return metrics