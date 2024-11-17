from huggingface_hub import HfApi, login  
import os
from pathlib import Path
from all_defs import DATASET_REPO_ID , MODEL_DATASET_DIR  , REPO_ID , \
    download_from_gdrive , clone_train_directory  , \
    MODEL_LOCAL_DIR , models_download_from_drive , \
    models_upload_to_dataset , DRIVE_MODELS_URL
from environs import Env


#clone_train_directory(15)

env = Env()
env.read_env()  # This loads the .env file

# Access the secret
api_token = os.getenv("MY_TOKEN")
# Log in to Hugging Face Hub
login(token=api_token)

files_to_upload = [
    { "../../pyproject.toml": "./pyproject.toml"},
    {"../../requirements.txt": "./requirements.txt"},    
    {"../models/train_model.py": "models/train_model.py"},
    {"./haggingface_model.py": "./haggingface_model.py"},
    {"./TensorboardStats.py": "./TensorboardStats.py"},
    {"./app.py": "./app.py"},
    {"./__init__.py": "./__init__.py"},
    {"./all_defs.py": "./all_defs.py"},
    #{"./data/train19": "./data/train19"}
]

# Upload the files to the repo
api = HfApi()
for item  in files_to_upload:
    file, path = list(item.items())[0]
    file_path = Path(file)
    
    if file_path.is_dir():
        for subdir, _, subfiles in os.walk(file_path):
            for subfile in subfiles:
                subfile_path = Path(subdir) / subfile
                relative_path = subfile_path.relative_to(file_path.parent)
                api.upload_file(
                    path_or_fileobj=subfile_path,
                    path_in_repo=str(relative_path),
                    repo_id=REPO_ID,
                    repo_type="space"
                )
                print(f"Uploaded {subfile_path} to {REPO_ID}")
    else:
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
files = [Path(file).name for file in files]

for model_file in models_upload_to_dataset:    
    if model_file in files:
        print(f"{model_file} exists in the repo.")
        print(f"Will not upload {model_file} to {MODEL_DATASET_DIR} on {DATASET_REPO_ID}")
    else:
        print(f"{model_file} does not exist in the repo.")
        print(f"Uploading {model_file} to {MODEL_DATASET_DIR} on {DATASET_REPO_ID}")


        # Check if the model file exists in the local directory
        model_file_path = Path(MODEL_LOCAL_DIR) / model_file
        print(f"Checking if {model_file} exists in the local directory {MODEL_LOCAL_DIR}")
        if not model_file_path.exists():
            print(f"{model_file} does not exist in the local directory {MODEL_LOCAL_DIR}. Checking google drive.")
            # Download the model from Google Drive
            if model_file in models_download_from_drive:
                print(f"Downloading {model_file} from Google Drive")
                download_from_gdrive(DRIVE_MODELS_URL, model_file)
                if not model_file_path.exists():
                    raise FileNotFoundError(f"Model file {model_file} not found in {MODEL_LOCAL_DIR}")
            else:
                print(f"Model file {model_file} not found in {models_download_from_drive} , skipping upload.")
                continue

        api.upload_file(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",    
            path_or_fileobj=model_file_path,
            path_in_repo=f"{MODEL_DATASET_DIR}/{model_file}"
)


