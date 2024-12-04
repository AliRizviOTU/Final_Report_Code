import os #Provides functions for interacting with the operating system, such as creating directories and working with file paths.
from sklearn.model_selection import train_test_split #This function from scikit-learn used to split datasets into subsets.
import shutil #utility library for file operations like copying and moving files.

# Setting paths for dataset and the target directories
base_dir = 'C:/Users/AliRi/Desktop/t/Uni/IMCS/Pics/raw-img'  #Path to the directory where the raw images are stored
target_dir = 'C:/Users/AliRi/Desktop/t/Uni/IMCS/split-data' # Path where the split datasets (training, validation, and test sets) will be saved.

# Create subdirectories for training, validation, and test sets
# Check if directories already exist to avoid recreating them
if not os.path.exists(target_dir):
    os.makedirs(os.path.join(target_dir, 'train/farfalla'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'train/ragno'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'val/farfalla'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'val/ragno'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'test/farfalla'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'test/ragno'), exist_ok=True)

# Get all image paths for butterfly (farfalla) and spider (ragno)
farfalla_images = [os.path.join(base_dir, 'farfalla', img) for img in os.listdir(os.path.join(base_dir, 'farfalla'))]
ragno_images = [os.path.join(base_dir, 'ragno', img) for img in os.listdir(os.path.join(base_dir, 'ragno'))]

# Split each class into train (80%), validation (10%), and test (10%) sets
farfalla_train, farfalla_temp = train_test_split(farfalla_images, test_size=0.2, random_state=42)
farfalla_val, farfalla_test = train_test_split(farfalla_temp, test_size=0.5, random_state=42)

ragno_train, ragno_temp = train_test_split(ragno_images, test_size=0.2, random_state=42)
ragno_val, ragno_test = train_test_split(ragno_temp, test_size=0.5, random_state=42)

# Helper function to copy files to the respective target directories
def copy_files(file_list, target_dir):
    for file in file_list:
        shutil.copy(file, target_dir)

# Copy butterfly (farfalla) and spider (ragno) images into train, val, and test directories
copy_files(farfalla_train, os.path.join(target_dir, 'train/farfalla'))
copy_files(farfalla_val, os.path.join(target_dir, 'val/farfalla'))
copy_files(farfalla_test, os.path.join(target_dir, 'test/farfalla'))

copy_files(ragno_train, os.path.join(target_dir, 'train/ragno'))
copy_files(ragno_val, os.path.join(target_dir, 'val/ragno'))
copy_files(ragno_test, os.path.join(target_dir, 'test/ragno'))

# Verify files were moved correctly
print(f"Training butterfly images: {len(os.listdir(os.path.join(target_dir, 'train/farfalla')))}")
print(f"Validation butterfly images: {len(os.listdir(os.path.join(target_dir, 'val/farfalla')))}")
print(f"Test butterfly images: {len(os.listdir(os.path.join(target_dir, 'test/farfalla')))}")

print(f"Training spider images: {len(os.listdir(os.path.join(target_dir, 'train/ragno')))}")
print(f"Validation spider images: {len(os.listdir(os.path.join(target_dir, 'val/ragno')))}")
print(f"Test spider images: {len(os.listdir(os.path.join(target_dir, 'test/ragno')))}")