import pyagrum as gum
import pyagrum.skbn as skbn
import pandas as pd
import numpy as np
import time
from setup import VERSION
import os, time
from utils_exp import load_data, calculate_weights, classification_metrics, split_into_classes

# Can choose something convenient below
save_path = "results\\NaiveBayes\\"
os.makedirs(save_path, exist_ok=True)

save_file = os.path.join(save_path,f"{VERSION}_test.txt")
open(save_file,'w')

metadata   = ['year', 'duration_ms']
physical   = ['loudness','tempo','instrumentalness','speechiness','liveness']
perceptual = ['danceability','energy','valence','acousticness']
structural = ['key','mode','time_signature']

if VERSION=='popularity':
    configs = [
        {"target":"popularity",
        "non-target":"genre",
        "num_classes":3},
        {"target":"popularity",
        "non-target":"genre",
        "num_classes":5}
    ]
elif VERSION=='genre':
    configs = [
        {"target":"genre",
        "non-target":"popularity",
        "num_classes":5}
    ]

subsets = [ physical, perceptual, structural]
subset_names = ["physical","perceptual","structural","metadata"]

for idx in range(len(configs)):
    config = configs[idx]
    target, num_classes = config["target"], config["num_classes"]
    
    DISCREET_FIELDS = ['danceability', 'energy', 'loudness', 
               'speechiness', 'acousticness', 'instrumentalness', 'liveness',
               'valence', 'tempo', 'duration_ms', 'year']
    NUM_CLASSES = {}
    for field in DISCREET_FIELDS:
        NUM_CLASSES[field] = num_classes

    subsets.append(metadata+[config["non-target"]])
    print(f"SUBSETS = {subsets}")
    for subset_idx in range(len(subsets)):
        subset, subset_name = subsets[subset_idx], subset_names[subset_idx]
        
        if VERSION=="popularity":
            train_data = pd.read_csv(f"train.csv").dropna()
            test_data = pd.read_csv(f"test.csv").dropna()
        elif VERSION=="genre":
            train_data = pd.read_csv(f"train_genre.csv").dropna()
            test_data = pd.read_csv(f"test_genre.csv").dropna()
        
        for key,value in NUM_CLASSES.items():
            if not key in subset:
                continue
            
            combined = pd.concat([train_data[key], test_data[key]], ignore_index=True)
            bin_edges = np.linspace(combined.min()-0.001, combined.max()+0.001, value + 1)
                    
            train_data[key] = np.digitize(train_data[key], bins=bin_edges, right=False) - 1
            test_data[key] = np.digitize(test_data[key], bins=bin_edges, right=False) - 1

        classifier = skbn.BNClassifier(learningMethod="NaiveBayes",prior='Smoothing')
        X_train = train_data[subset].to_numpy()[:,:]
        y_train = train_data[target].to_numpy(dtype=np.int32)
        if VERSION=="popularity":
            y_train = split_into_classes(y_train, num_classes)

        classifier.fit(X=X_train, y=y_train)  # target column

        X_test = test_data[subset].to_numpy()
        y_test = test_data[target].to_numpy(dtype=np.int32)
        if VERSION=="popularity":
            y_test = split_into_classes(y_test, num_classes)

        start = time.time_ns()
        
        classifier.fit(X=X_train, y=y_train) 
        y_pred = classifier.predict(X_test)

        train_time = (time.time_ns()-start)*(10**-9)
        metrics = classification_metrics(y_test, y_pred)
        
        with open(save_file,'a') as f:
            f.write(f"{num_classes} {subset_name}\n")
            f.write( str(list(metrics.values()) + [train_time]) + "\n" )
