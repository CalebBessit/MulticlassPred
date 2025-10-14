# Creates and trains an MLP and XGBoost model
# Caleb Bessit
# 14 October 2025

import time
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

from utils import load_data, classification_metrics


X_train, y_train = load_data("train")
X_test, y_test   = load_data("test")

xgb = XGBClassifier(random_state=0)
mlp = MLPClassifier(random_state=0, max_iter=1000)

#Used for XGBoost, because model expects labels starting from zero (which we have removed)
le = LabelEncoder()

# XGBoost
print(f"Training and testing XGBoost model...")
start = time.time_ns()
y_train_encoded = le.fit_transform(y_train)
xgb.fit(X_train, y_train_encoded)
y_pred_encoded = xgb.predict(X_test)
y_pred = le.inverse_transform(y_pred_encoded)


classification_metrics(y_test, y_pred)

print(f"XGBoost took {(time.time_ns()-start)*(10**-9)} seconds to train.")

# MLP
print(f"\nTraining and testing MLP...")
start = time.time_ns()
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

classification_metrics(y_test, y_pred)

print(f"MLP took {(time.time_ns()-start)*(10**-9)} seconds to train.")
