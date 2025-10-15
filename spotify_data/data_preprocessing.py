# Preprocess Spotify data by replacing text with values and split into train/validate/split sets
# Caleb Bessit
# 14 October 2025

import pandas as pd
from setup import VERSION

if VERSION=="popularity":
    data = pd.read_csv("spotify_data.csv")
elif VERSION=="genre":
    data = pd.read_csv("genre_labelled_data.csv")

len_train = int(0.8*len(data))
len_test  = len(data) - len_train 


data_shuffled = data.sample(frac=1, random_state=0)

train = data_shuffled[:len_train]
test  = data_shuffled[len_train:]

if VERSION=="popularity":
    train.to_csv("train.csv")
    test.to_csv("test.csv")
elif VERSION=="genre":
    train.to_csv("train_genre.csv")
    test.to_csv("test_genre.csv")

print(f"Doing {VERSION}.")
print(f"Saved train data: {len(train)} rows.")
print(f"Saved test data: {len(test)} rows.")


