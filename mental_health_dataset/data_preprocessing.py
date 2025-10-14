# Preprocess mental health data by replacing text with values and split into train/validate/split sets
# Caleb Bessit
# 14 October 2025

import pandas as pd

data = pd.read_csv("mxmh_survey_results_values.csv")

# print(len(data))

len_train = int(0.8*len(data))
# len_val   = int(0.1*len(data))
len_test  = len(data) - len_train #- len_val

data_shuffled = data.sample(frac=1)

train = data[:len_train]
# val   = data[len_train: len_train+len_val]
test  = data[len_train:]

train.to_csv("train.csv")
print(f"Saved train data: {len(train)} rows.")
# val.to_csv("val.csv")
test.to_csv("test.csv")
print(f"Saved test data: {len(test)} rows.")


