# Data analysis of spotify data
# Caleb Bessit
# 15 Octobet 2025

import pandas as pd
import matplotlib.pyplot as plt

from utils import split_into_classes

data = pd.read_csv("spotify_data.csv")
data = data.dropna()

top_five_genres = ['acoustic','ambient','black-metal','gospel','alt-rock']
replacements = {genre:top_five_genres.index(genre) for genre in top_five_genres}
print(replacements)
filtered_data = data[data['genre'].isin(top_five_genres)]

for genre, index in replacements.items():
    filtered_data['genre'] = filtered_data['genre'].replace(genre, index)

# filtered_data['genre'] = filtered_data['genre'].replace(replacements)

filtered_data.to_csv("genre_labelled_data.csv")
# print(filtered_data['genre'])
# print(len(data), len(filtered_data))

# counts = data['genre'].value_counts(ascending=False)

# counts.plot(kind='barh')
# plt.xlabel("Number of occurrances")
# plt.ylabel("Genres")
# plt.title("Proportion of genres in dataset")
# plt.show()
