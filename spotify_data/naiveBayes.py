import pyagrum as gum
import pyagrum.lib.notebook as gnb
import pyagrum.lib.image as gumimage
import pandas as pd
import numpy as np
from utils import load_data, classification_metrics, calculate_weights, split_into_classes
from setup import VERSION
import pyagrum.skbn as skbn
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import csv

NUM_POP_CLASSES = 3
DISCREET_FIELDS = ['danceability', 'energy', 'loudness', 
               'speechiness', 'acousticness', 'instrumentalness', 'liveness',
               'valence', 'tempo', 'duration_ms', 'year']
NUM_CLASSES = {}
for field in DISCREET_FIELDS:
    NUM_CLASSES[field] = NUM_POP_CLASSES


if VERSION=="popularity":
    train_data = pd.read_csv(f"train.csv").dropna()
    test_data = pd.read_csv(f"test.csv").dropna()
elif VERSION=="genre":
    train_data = pd.read_csv(f"train_genre.csv").dropna()
    test_data = pd.read_csv(f"test_genre.csv").dropna()
    

if VERSION=="popularity":
    FEATURES = [ 'year', 'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo', 'duration_ms', 'time_signature']

    TARGET = ["popularity"]
elif VERSION=="genre":
    FEATURES = [ 'year', 'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo', 'duration_ms', 'time_signature','popularity']

    TARGET = ["genre"]
    
    NUM_CLASSES['popularity'] = NUM_POP_CLASSES


results_file = f"naive_pop_{NUM_POP_CLASSES}.csv"
# with open(results_file, "w", newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(list(NUM_CLASSES.keys()))


for key,value in NUM_CLASSES.items():
    combined = pd.concat([train_data[key], test_data[key]], ignore_index=True)
    bin_edges = np.linspace(combined.min()-0.001, combined.max()+0.001, value + 1)
            
    train_data[key] = np.digitize(train_data[key], bins=bin_edges, right=False) - 1
    test_data[key] = np.digitize(test_data[key], bins=bin_edges, right=False) - 1                
for feat in FEATURES:
    if not feat in NUM_CLASSES.keys():
        continue 
    for nc in [NUM_POP_CLASSES]:
        NUM_CLASSES[feat] = nc
        combined = pd.concat([train_data[feat], test_data[feat]], ignore_index=True)
        bin_edges = np.linspace(combined.min()-0.001, combined.max()+0.001, nc + 1)
            
        train_data[feat] = np.digitize(train_data[feat], bins=bin_edges, right=False) - 1
        test_data[feat] = np.digitize(test_data[feat], bins=bin_edges, right=False) - 1 
        
        prior = {}
        for i in combined:
            prior[i] = 1

        classifier = skbn.BNClassifier(learningMethod="NaiveBayes",prior='Smoothing')
        x_train = train_data[FEATURES].to_numpy()[:,:]
        y_train = train_data[TARGET].to_numpy(dtype=np.int32)[:,0]
        if VERSION=="popularity":
            y_train = split_into_classes(y_train, NUM_POP_CLASSES)

        classifier.fit(X=x_train, y=y_train)  # target column

        x_test = test_data[FEATURES].to_numpy()[:,:]
        y_test = test_data[TARGET].to_numpy(dtype=np.int32)[:,0]
        if VERSION=="popularity":
            y_test = split_into_classes(y_test, NUM_POP_CLASSES)

        y_pred = classifier.predict(x_test)

        accuracy = accuracy_score(y_test,y_pred)
        baccuracy = balanced_accuracy_score(y_test,y_pred)

        gumimage.export(classifier.bn, f"network_{VERSION}.pdf")

        with open(results_file, "a", newline='\n') as f:
            row = list(NUM_CLASSES.values())
            row.extend([accuracy,baccuracy])
            writer = csv.writer(f)
            writer.writerow(row)
    NUM_CLASSES[feat] = NUM_POP_CLASSES
    combined = pd.concat([train_data[feat], test_data[feat]], ignore_index=True)
    bin_edges = np.linspace(combined.min()-0.001, combined.max()+0.001, NUM_POP_CLASSES + 1)
            
    train_data[feat] = np.digitize(train_data[feat], bins=bin_edges, right=False) - 1
    test_data[feat] = np.digitize(test_data[feat], bins=bin_edges, right=False) - 1 
