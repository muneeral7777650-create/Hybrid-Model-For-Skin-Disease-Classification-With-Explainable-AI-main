# ============================================================
# check_data.py
# ------------------------------------------------------------
# Purpose:
# Before running the main training code, this script verifies
# that all the necessary dataset files (metadata CSV and images)
# are correctly available and formatted.
# ============================================================

# Import required libraries
import os       # for checking if files/folders exist
import pandas as pd   # for reading and inspecting the CSV file

# ------------------------------------------------------------
# STEP 1: Define dataset paths
# ------------------------------------------------------------
# Change these paths to match the actual location of your dataset.
# The raw string (r"...") ensures Windows backslashes (\) work correctly.
# DATA_DIR = r'path_to_your_dataset\HAM10000_metadata.csv'         # Metadata CSV file
# IMG_DIR_1 = r'path_to_your_dataset\HAM10000_images_part_1'       # First folder of images
# IMG_DIR_2 = r'path_to_your_dataset\HAM10000_images_part_2'       # Second folder of images
DATA_DIR = r"E:\00. Master's in AI\Sem 1\AI\Project\code new\archive\HAM10000_metadata.csv"
IMG_DIR_1 = r"E:\00. Master's in AI\Sem 1\AI\Project\code new\archive\HAM10000_images_part_1"
IMG_DIR_2 = r"E:\00. Master's in AI\Sem 1\AI\Project\code new\archive\HAM10000_images_part_2"
# ------------------------------------------------------------
# STEP 2: Check if the metadata CSV file exists
# ------------------------------------------------------------
# os.path.exists(path) → returns True if the file/folder exists, else False.
if not os.path.exists(DATA_DIR):
    # If file not found, print a warning message with the missing path
    print(f"Metadata file not found: {DATA_DIR}")
else:
    # Otherwise, confirm the metadata file is present
    print("✅ Metadata file found!")

# ------------------------------------------------------------
# STEP 3: Check if the image folders exist
# ------------------------------------------------------------
# We'll loop through both folder paths and check each.
missing_folders = []  # list to store names of missing folders

for folder in [IMG_DIR_1, IMG_DIR_2]:
    # If folder doesn't exist, add it to the missing list
    if not os.path.exists(folder):
        missing_folders.append(folder)

# Print results depending on whether folders are missing or not
if missing_folders:
    print(f"⚠️ Missing image folders: {missing_folders}")
else:
    print("✅ All image folders found!")

# ------------------------------------------------------------
# STEP 4: Check dataset consistency (CSV structure)
# ------------------------------------------------------------
# Read the metadata CSV file using pandas
# If the file path is wrong, this line will raise a FileNotFoundError.
data = pd.read_csv(DATA_DIR)

# Check if essential columns ('image_id' and 'dx') are present in the CSV
if 'image_id' in data.columns and 'dx' in data.columns:
    print("✅ Metadata structure is correct!")
else:
    print("❌ Metadata structure is incorrect. Check CSV format.")

# ------------------------------------------------------------
# END OF SCRIPT
# ------------------------------------------------------------
# Expected Output (if everything is correct):
# ✅ Metadata file found!
# ✅ All image folders found!
# ✅ Metadata structure is correct!
# ------------------------------------------------------------
