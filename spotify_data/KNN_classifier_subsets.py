import os, time
import numpy as np
from setup import VERSION
from sklearn.neighbors import KNeighborsClassifier
from utils_exp import load_data, classification_metrics

# Can choose something convenient below
save_path = "results/KNN/"
os.makedirs(save_path, exist_ok=True)

# neighbours = [1,2,3,4,5,10,20,50,100,250,500,1000,5000]
neighbours = [1,2,3]

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

    subsets.append(metadata+[config["non-target"]])
    for subset_idx in range(len(subsets)):
        subset, subset_name = subsets[subset_idx], subset_names[subset_idx]
        
        X_train, y_train = load_data("train",num_classes,subset,target) if target=="popularity" else load_data("train_genre",num_classes,subset,target)
        X_test, y_test   = load_data("test",num_classes,subset,target) if target=="popularity" else load_data("test_genre",num_classes,subset,target)

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
        np.save(  os.path.join(save_path, f"knn_{VERSION}_{num_classes}_{subset_name}.npy"), np.array(results) )
    subsets.pop()
