from pathlib import Path
import os
import gdown
# origanization : MLSHYAorg

REPO_ID = "lmanov1/timm-efficientnet_b3.ra2_in1k"
DATASET_REPO_ID = "lmanov1/BUI17_data"
MODEL_LOCAL_DIR = Path("../../data/output/")

# Upload the models to the repo (dataset)
models_upload_to_dataset = [    
    # "pretrained_trainLabels19_export.pth",
    # "pretrained_trainLabels19_export.pkl",
    # "v1.0_vgg16_model.keras",
    "v4.6_efficientnet-b7_model.pth",
    "v4.6.2_efficientnet-b7_model.pth",

    ]
MODEL_DATASET_DIR = Path("models/")

# Download the models from Google Drive
models_download_from_drive = [
    #"v1.0_vgg16_model.keras",
    "v4.6.2/v4.6.2_efficientnet-b7_model.pth"
]

#DRIVE_MODELS_URL = "https://drive.google.com/file/d/1Y1XBp5m-AIc_eBGwQPcOjOJRiirPB9mA/view?usp=sharing"
# Models
DRIVE_MODELS_URL = "https://drive.google.com/drive/folders/1gLZlJRswvHyvzNiHxmjjQpDolMGduXcy?usp=sharing"
#https://drive.google.com/file/d/14281aO0pl24FAl5wbhiAsg-KexWoHBnK/view?usp=sharing

# Classes vocabilary for Hugging face GUI
classes_mapping = [ {0: "No DR"}, {1: "Mild DR"}, {2: "Moderate DR"}, {3: "Severe DR"}, {4: "Proliferative DR"}]    
def translate_labels(values):
    print(values)
    labels = []
    for value in values:
        value = int(value)
        label = classes_mapping[value]
        #label = next((label for d in classes_mapping for k, label in d.items() if k == value), None)
        if label is not None:
            labels.append(label)
    print(labels)
    return labels


def download_from_gdrive(url, filename):
    # Extract the file ID from the URL
    file_id = url.split('/')[-2]
    download_url = f"https://drive.google.com/uc?id={file_id}"
    import urllib.parse
    print(f"download_from_gdrive:Downloading file from Google Drive @ {url} => {filename}")
    # URL encode the download URL
    download_url = urllib.parse.quote(download_url, safe=':/?=&')
    # Download the file
    
    local_filename = os.path.basename(filename)
    output = f"{MODEL_LOCAL_DIR}/{local_filename}"
    print(f"Downloading file from Google Drive @ {url} => {filename} into {output}")
    if Path(output).exists():
        print(f"File '{output}' already exists. Skipping download.")
    else:
        gdown.download(url, output, quiet=False, fuzzy=True)
        print(f"File downloaded to: {output}")


from pathlib import Path
import shutil
import random

# Function to clone the directory structure from ../../notebooks/data/train19 to ./data/train19
# to upload to Hugging Face Hub
def clone_train_directory(number_of_files: int = -1):
    """
    Clone directory structure from ../../notebooks/data/train19 to ./data/train19
    Args:
        number_of_files: Number of files to copy from each subdirectory. -1 means copy all
    """
    # Setup source and target paths
    source_dir = Path('../../notebooks/data/train19')
    target_dir = Path('./data/train19')
    
    if not source_dir.exists():
        raise ValueError(f"Source directory {source_dir} does not exist")
        
    # Create target root if needed
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Walk through all subdirectories
    for source_subdir in source_dir.glob('**/'):
        # Create relative path to maintain structure
        rel_path = source_subdir.relative_to(source_dir)
        target_subdir = target_dir / rel_path
        
        # Create target subdirectory
        target_subdir.mkdir(parents=True, exist_ok=True)
        
        # Get all files in current subdirectory
        files = list(source_subdir.glob('*.*'))
        
        # Skip if no files or this is the root directory
        if not files or source_subdir == source_dir:
            continue            
        # Select files to copy
        if number_of_files == -1:
            files_to_copy = files
        else:
            files_to_copy = random.sample(files, min(number_of_files, len(files)))
            
        # Copy selected files
        for file in files_to_copy:
            target_file = target_subdir / file.name
            shutil.copy2(file, target_file)
            
    print(f"Directory structure cloned with {number_of_files} files per subdirectory")

def print_directory_recursively(path):
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")