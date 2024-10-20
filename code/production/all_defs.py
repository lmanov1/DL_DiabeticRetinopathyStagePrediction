from pathlib import Path
import os
import gdown
# origanization : MLSHYAorg

REPO_ID = "lmanov1/timm-efficientnet_b3.ra2_in1k"
DATASET_REPO_ID = "lmanov1/BUI17_data"
MODEL_LOCAL_DIR = Path("../../data/output/")

# Upload the models to the repo (dataset)
models_upload_to_dataset = [    
    "pretrained_trainLabels19_export.pth",
    "pretrained_trainLabels19_export.pkl",
    "v1.0_vgg16_model.keras"
    ]
MODEL_DATASET_DIR = Path("models/")

# Download the models from Google Drive
models_download_from_drive = [
    "v1.0_vgg16_model.keras"
]

DRIVE_MODELS_URL = "https://drive.google.com/file/d/1Y1XBp5m-AIc_eBGwQPcOjOJRiirPB9mA/view?usp=sharing"



# Classes vocabilary for Hugging face GUI
classes_mapping = [ {0: "No DR"}, {1: "Mild DR"}, {2: "Moderate DR"}, {3: "Severe DR"}, {4: "Proliferative DR"}]    
def translate_labels(values):
    labels = []
    for value in values:
        label = next((label for d in classes_mapping for k, label in d.items() if k == value), None)
        if label is not None:
            labels.append(label)
    print(labels)
    return labels


def download_from_gdrive(url, filename):
    # Extract the file ID from the URL
    file_id = url.split('/')[-2]
    download_url = f"https://drive.google.com/uc?id={file_id}"
    import urllib.parse

    # URL encode the download URL
    download_url = urllib.parse.quote(download_url, safe=':/?=&')
    # Download the file
    
    output = f"{MODEL_LOCAL_DIR}/{filename}"
    print(f"Downloading file from Google Drive @ {url} => {filename} into {output}")
    if Path(output).exists():
        print(f"File '{output}' already exists. Skipping download.")
    else:
        gdown.download(url, output, quiet=False, fuzzy=True)
        print(f"File downloaded to: {output}")
