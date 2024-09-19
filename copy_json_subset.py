import shutil
import os

# Path to the list of JSON files
json_list_file = "/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/datasets/greglist.txt"

# Source directory containing all JSON files
source_dir = "/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/datasets/2020 dataset/training"

# Destination directory for copied files
dest_dir = "/Users/vitsiozo/Desktop/MSc AI/Modules/Project/ARC/datasets/2020 dataset/subset50"

# Create the destination directory if it doesn't exist
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Read the list of JSON files from the text file
with open(json_list_file, 'r') as f:
    json_files = f.read().splitlines()

# Copy the specified JSON files to the destination directory
for json_file in json_files:
    source_path = os.path.join(source_dir, json_file)
    dest_path = os.path.join(dest_dir, json_file)
    shutil.copy(source_path, dest_path)