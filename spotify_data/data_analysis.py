# Data analysis of spotify data
# Caleb Bessit
# 15 Octobet 2025

import pandas as pd
import numpy as np
import matplot2tikz
from collections import Counter
import matplotlib.pyplot as plt

from utils import split_into_classes

data = pd.read_csv("spotify_data.csv")
data = data.dropna()

# top_five_genres = ['acoustic','ambient','black-metal','gospel','alt-rock']
top_10 = ['acoustic','ambient','black-metal','gospel','alt-rock','emo','indian','k-pop','new-age','blues']
bott_10 = ['metalcore','pop','punk','sad','house','chicago-house','dubstep','detroit-techno','rock','songwriter']





# replacements = {genre:top_five_genres.index(genre) for genre in top_five_genres}
# print(replacements)
# filtered_data = data[data['genre'].isin(top_five_genres)]

# for genre, index in replacements.items():
#     filtered_data['genre'] = filtered_data['genre'].replace(genre, index)

# # filtered_data['genre'] = filtered_data['genre'].replace(replacements)

# filtered_data.to_csv("genre_labelled_data.csv")
# print(filtered_data['genre'])
# print(len(data), len(filtered_data))

counts = data['genre'].value_counts(ascending=False)

# counts= data[data['genre'].isin(top_five_genres)]['genre'].value_counts()
top_counts= data[data['genre'].isin(top_10)]['genre'].value_counts(ascending=False)
bott_counts= data[data['genre'].isin(bott_10)]['genre'].value_counts(ascending=False)


combined = pd.concat([top_counts, bott_counts])
# combined = combined[::-1]

# Define colors and textures
colors = ['orange'] * 10 + ['skyblue'] * 10
hatches = [''] * 10 + ['//'] * 10  # texture for bottom 10

# Create plot
fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(combined.index, combined.values, color=colors, edgecolor='black')

# Apply hatching
for i, bar in enumerate(bars):
    bar.set_hatch(hatches[i])

# Add labels
for i, (value, bar) in enumerate(zip(combined.values, bars)):
    ax.text(value + max(combined.values)*0.01,
            bar.get_y() + bar.get_height()/2,
            str(value),
            va='center', ha='left', fontsize=10)

# Axis labels and title
ax.set_xlabel('Count', fontsize=12)
ax.set_xscale('log')
ax.set_ylabel('Genre', fontsize=12)
ax.set_title('Number of Songs for the 10 most and least common genres', fontsize=14, pad=15)
ax.invert_yaxis()

plt.tight_layout()
plt.grid(alpha=0.3)
matplot2tikz.save('figures/most_and_least_genres.tex')
plt.show()


print()
print(f"Top 10: {counts[:10]}\n\nBottom 10: {counts[-10:]}")
# counts.plot(kind='barh', color='salmon', edgecolor='k')
# plt.xlabel("Number of occurrances")
# plt.title("Number of samples of top 5 genres in dataset")

# matplot2tikz.save("figures/top_five_genres.tex")
# plt.show()

############################################## Popularity stuff

# three_class_pop = split_into_classes(data['popularity'].to_numpy(np.int32), 3)

# five_class_pop = split_into_classes(data['popularity'].to_numpy(), 5)

# three_pos, five_pos = np.arange(3), np.arange(5)



# three_bounds = [0,33,66,100]
# five_bounds = [0,20,40,60,80,100]


# three_classes = dict(Counter(three_class_pop))
# five_classes = dict(Counter(five_class_pop))


# Plotting three classes
# plt.figure()
# # plt.ylim(bottom=1.2*three_max)
# for i in range(3):
#     plt.bar( three_pos[i], three_classes[i],0.4, color='royalblue',alpha= ( (three_pos[i]+1)/3),edgecolor='black', )
#     plt.text(three_pos[i], 1.1*three_classes[i], f"{three_classes[i]}", ha='center')

# three_labels = []
# for i in range(3):

#     three_labels.append(f"{i} - ({three_bounds[i]},{three_bounds[i+1]}]")

# plt.xticks(three_pos, three_labels)
# plt.xlabel("Class label")
# plt.ylabel("Occurrances")

# plt.title("Distribution of popularity scores with three classes")
# plt.grid(alpha=0.3)
# plt.yscale("log")
# matplot2tikz.save("figures/three_distrib.tex")
# plt.show()

# FIve classes

# plt.figure("Five classes")

# five_labels = []
# for i in range(5):
#     five_labels.append(f"{i} - ({five_bounds[i]},{five_bounds[i+1]}]")

# for i in range(5):
#     plt.bar( five_pos[i], five_classes[i],0.4, color='green',alpha= ( (five_pos[i]+1)/5),edgecolor='black', )
#     plt.text(five_pos[i], 1.1*five_classes[i], f"{five_classes[i]}", ha='center')
# plt.xticks( five_pos, five_labels)
# plt.xlabel("Class label")
# plt.ylabel("Occurrances")

# plt.title("Distribution of popularity scores with five classes")
# plt.grid(alpha=0.3)
# plt.yscale("log")
# matplot2tikz.save("figures/five_distrib.tex")
# plt.show()

# print(three_classes)

# print(f"\n{five_classes}")
