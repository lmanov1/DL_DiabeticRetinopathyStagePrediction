from huggingface_hub import HfApi, login  
import os
from pathlib import Path
from all_defs import DATASET_REPO_ID , MODEL_DATASET_PATH  , REPO_ID , MODEL_LOCAL_FILE_PATH

# Log in to Hugging Face
MY_TOKEN = "MY TOKEN TP PUT HERE EEEE"
login(token=MY_TOKEN)

files_to_upload = [
    { "../../pyproject.toml": "./pyproject.toml"},
    {"../../requirements.txt": "./requirements.txt"},    
    {"../models/train_model.py": "models/train_model.py"},
    {"./app.py": "./app.py"},
    {"./__init__.py": "./__init__.py"},
    {"./all_defs.py": "./all_defs.py"}
]

# Upload the files to the repo
api = HfApi()
for item  in files_to_upload:
    file, path = list(item.items())[0]
    file_path = Path(file)
    api.upload_file(
        path_or_fileobj=file,        
        path_in_repo=path,
        repo_id=REPO_ID,
        repo_type="space"
    )
    print(f"Uploaded {file_path.name} to {REPO_ID}")

print(f"Uploaded files to {REPO_ID}")
# List the files in the repo
files = api.list_repo_files(repo_id=DATASET_REPO_ID, repo_type="dataset")
print(f"Files in {DATASET_REPO_ID}: {files}")
# Check if the file already exists in the repo
model_file = Path(MODEL_DATASET_PATH).name
files = [Path(file).name for file in files]

if model_file in files:
    print(f"{model_file} exists in the repo.")
else:
    print(f"{model_file} does not exist in the repo.")
    print(f"Uploading {MODEL_LOCAL_FILE_PATH} to {MODEL_DATASET_PATH} on {DATASET_REPO_ID}")

    api.upload_file(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",    
        path_or_fileobj=MODEL_LOCAL_FILE_PATH,
        path_in_repo=str(MODEL_DATASET_PATH)
)


