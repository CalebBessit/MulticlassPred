# Splitting popularity into classes based on ranges
# Caleb Bessit
# 15 October 2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

data = pd.read_csv("train.csv")

pop = data["popularity"].to_numpy()

classes = 5
size = 100//classes

print(f"Class counts")
for i in range(classes):
    lower, upper = size*i, size*(i+1)
    if i == (classes-1):
        upper = 100

    mask = (pop>lower) & (pop<=upper)

    print(f"\t+ For {i}: range is ({lower}, {upper}], count is {np.sum(mask)}")

    pop[mask] = i

freqs = dict(Counter(pop))
print(freqs)

