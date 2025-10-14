# Utility script to load the data
# Caleb Bessit
# 14 October 2025

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

ORIGINAL_FEATURES = ["Age","Primary streaming service","Hours per day","While working","Instrumentalist",
            "Composer","Fav genre","Exploratory","Foreign languages","BPM","Frequency [Classical]",
            "Frequency [Country]","Frequency [EDM]","Frequency [Folk]","Frequency [Gospel]",
            "Frequency [Hip hop]","Frequency [Jazz]","Frequency [K pop]","Frequency [Latin]",
            "Frequency [Lofi]","Frequency [Metal]","Frequency [Pop]","Frequency [R&B]","Frequency [Rap]",
            "Frequency [Rock]","Frequency [Video game music]",
            "Anxiety","Depression","Insomnia","OCD","Music effects"]

FEATURES = ["Age","Hours per day","While working","Instrumentalist",
            "Composer","Exploratory","Foreign languages","BPM","Frequency [Classical]",
            "Frequency [Country]","Frequency [EDM]","Frequency [Folk]","Frequency [Gospel]",
            "Frequency [Hip hop]","Frequency [Jazz]","Frequency [K pop]","Frequency [Latin]",
            "Frequency [Lofi]","Frequency [Metal]","Frequency [Pop]","Frequency [R&B]","Frequency [Rap]",
            "Frequency [Rock]","Frequency [Video game music]","Music effects"]

TARGET = ["Anxiety"]



def load_data(fileset):
    print(f"Loading {fileset} data...")

    X, y = None, None

    data = pd.read_csv(f"{fileset}.csv")
    data = data.dropna()

    X = data[FEATURES].to_numpy()
    y = data[TARGET].to_numpy(dtype=np.int32)[:,0]

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