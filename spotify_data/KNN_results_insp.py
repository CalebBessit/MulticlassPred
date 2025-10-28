# Converts .npy result files produced by classifier.py into a format that is easier to examine
# Matthew Dean
# 16 October 2025

import numpy as np

neighbours = [1,2,3,4,5,10,20,50,100,250,500,1000,5000]

files = ["popularity_3", "popularity_5", "genre_5"]

for file in files:
    data = np.load(f"results\\KNN\\knn_{file}.npy", allow_pickle=True)
    with open(f"results\\KNN\\knn_{file}.txt",'w') as f:
        for key, value in enumerate(data):
            f.write(f"{neighbours[key]} : {str(value)}\n")