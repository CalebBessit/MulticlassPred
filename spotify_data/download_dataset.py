# Download 1 Million spotify track dataset from Kaggle
# Matthew Dean
# 15 October 2025

import kagglehub
import shutil
import os
sep = os.sep

folder_path = path = os.path.join(
    os.path.expanduser("~"),  # expands to current user's home directory
    ".cache",
    "kagglehub",
    "datasets",
    "amitanshjoshi"
)

if os.path.exists(folder_path):
    shutil.rmtree(folder_path)
    print("Folder and all contents deleted.")
else:
    pass



# Download the latest version
path = kagglehub.dataset_download("amitanshjoshi/spotify-1million-tracks")
path += sep+"spotify_data.csv"
print("Downloaded to:", path)

# Define where you want to move it
target_dir = "."

# Create the directory if it doesn’t exist
os.makedirs(target_dir, exist_ok=True)

# Move the dataset folder
shutil.move(path, target_dir)

print("Moved dataset to:", target_dir)