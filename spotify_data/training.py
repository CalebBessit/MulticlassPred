# Creates and trains an MLP and XGBoost model
# Caleb Bessit
# 14 October 2025

import time
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from utils import load_data, classification_metrics, calculate_weights


# Params
NUM_CLASSES = 3

X_train, y_train = load_data("train", NUM_CLASSES)
X_test, y_test   = load_data("test", NUM_CLASSES)

xgb = XGBClassifier(random_state=0)
mlp = MLPClassifier(random_state=0, max_iter=1000)

#Used for XGBoost, because model expects labels starting from zero (which we have removed)
le = LabelEncoder()

# XGBoost
print(f"Training and testing XGBoost model...")
start = time.time_ns()
y_train_encoded = le.fit_transform(y_train)
class_weights, sample_weights = calculate_weights(y_train_encoded)

xgb_pl = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", xgb)
])

xgb_pl.fit(X_train, y_train_encoded, xgb__sample_weight=sample_weights)
y_pred_encoded = xgb_pl.predict(X_test)
y_pred = le.inverse_transform(y_pred_encoded)


classification_metrics(y_test, y_pred)

print(f"XGBoost took {(time.time_ns()-start)*(10**-9)} seconds to train.")

# MLP
print(f"\nCalculating sample weights for MLP...")
class_weights, sample_weights = calculate_weights(y_train)
print(f"Training and testing MLP...")
start = time.time_ns()

mlp_pl = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", mlp)
])

# MLP does not directly support handling class imbalance. Have to calculate class weights and pass it in.

mlp_pl.fit(X_train, y_train, mlp__sample_weight=sample_weights)
y_pred = mlp_pl.predict(X_test)

classification_metrics(y_test, y_pred)

print(f"MLP took {(time.time_ns()-start)*(10**-9)} seconds to train.")


# Logistic regression model

print(f"\nTraining and testing Logistic regression model...")
start = time.time_ns()
pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(class_weight="balanced", random_state=0))
    ])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
classification_metrics(y_test, y_pred)

print(f"Logistic regression took {(time.time_ns()-start)*(10**-9)} seconds to train.")
