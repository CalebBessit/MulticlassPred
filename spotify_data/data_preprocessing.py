# Preprocess Spotify data by replacing text with values and split into train/validate/split sets
# Caleb Bessit
# 14 October 2025

import pandas as pd

data = pd.read_csv("spotify_data.csv")

# Will use later for genre classifications
genres = ['acoustic', 'afrobeat', 'alt-rock', 'ambient', 'black-metal',
       'blues', 'breakbeat', 'cantopop', 'chicago-house', 'chill',
       'classical', 'club', 'comedy', 'country', 'dance', 'dancehall',
       'death-metal', 'deep-house', 'detroit-techno', 'disco',
       'drum-and-bass', 'dub', 'dubstep', 'edm', 'electro', 'electronic',
       'emo', 'folk', 'forro', 'french', 'funk', 'garage', 'german',
       'gospel', 'goth', 'grindcore', 'groove', 'guitar', 'hard-rock',
       'hardcore', 'hardstyle', 'heavy-metal', 'hip-hop', 'house',
       'indian', 'indie-pop', 'industrial', 'jazz', 'k-pop', 'metal',
       'metalcore', 'minimal-techno', 'new-age', 'opera', 'party',
       'piano', 'pop', 'pop-film', 'power-pop', 'progressive-house',
       'psych-rock', 'punk', 'punk-rock', 'rock', 'rock-n-roll',
       'romance', 'sad', 'salsa', 'samba', 'sertanejo', 'show-tunes',
       'singer-songwriter', 'ska', 'sleep', 'songwriter', 'soul',
       'spanish', 'swedish', 'tango', 'techno', 'trance', 'trip-hop']

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


