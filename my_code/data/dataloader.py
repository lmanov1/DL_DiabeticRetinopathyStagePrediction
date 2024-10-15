import os
import subprocess
import sys
import platform
import zipfile
import json

from kaggle.api.kaggle_api_extended import KaggleApi
from my_code.Util.MultiPlatform import standardize_path


def update_Global_os_param():
    """Add the project root directory to the Python path."""
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    print(f"{platform.system()} platform detected. - Global OS parameters updated")


update_Global_os_param()


class Dataloader:
    def __init__(self, dataset_path, dataset_name, kaggle_json_path=None, batch_size=5):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.dataset_dir = os.path.join(self.dataset_path, self.dataset_name)
        self.kaggle_json_path = kaggle_json_path
        self.batch_size = batch_size
        self.kaggle_username = None
        self.kaggle_key = None

        print(f"Dataset path: {self.dataset_path}, Dataset: {self.dataset_name}, Kaggle JSON path: {self.kaggle_json_path}")

        self.create_kaggle_json()
        self.setup_kaggle_api()

    def set_credentials(self, username, key):
        """Set Kaggle API credentials."""
        self.kaggle_username = username
        self.kaggle_key = key

        # Save credentials in a kaggle.json file
        kaggle_json_content = {
            'username': username,
            'key': key
        }

        os.makedirs(os.path.dirname(self.kaggle_json_path), exist_ok=True)
        with open(self.kaggle_json_path, 'w') as f:
            json.dump(kaggle_json_content, f)
        print(f"Kaggle credentials set and saved to {self.kaggle_json_path}")

    def create_kaggle_json(self):
        """Create the Kaggle API JSON file if it doesn't exist."""
        if self.kaggle_json_path is None:
            home_directory = os.path.expanduser("~")
            self.kaggle_json_path = standardize_path(home_directory, '.kaggle', 'kaggle.json')

        if not os.path.exists(self.kaggle_json_path):
            raise FileNotFoundError(f"Kaggle JSON file not found at {self.kaggle_json_path}")

    def setup_kaggle_api(self):
        """Setup Kaggle API using the JSON file."""
        try:
            self.api = KaggleApi()
            self.api.authenticate()  # Ensure you are authenticated
            print("Kaggle API authenticated successfully.")

            # Check if the dataset is already downloaded
            if os.path.isdir(self.dataset_dir):
                print(f"Dataset {self.dataset_name} was already downloaded.")
                return

            print(f"Dataset {self.dataset_dir} not found - downloading dataset {self.dataset_name}")
            self.download_dataset_in_batches()

        except Exception as e:
            print(f"An error occurred during Kaggle API setup: {e}")
            import traceback
            traceback.print_exc()

    def download_dataset_in_batches(self):
        """Download dataset files in batches."""
        print(f"Retrieving list of files for dataset: {self.dataset_name}")
        # Get the list of files in the dataset
        dataset_files = self.api.dataset_list_files(self.dataset_name)
        file_names = [file.name for file in dataset_files.files]  # Extract file names
        print(f"Files found: {file_names}")

        # Download files in batches
        for i in range(0, len(file_names), self.batch_size):
            batch_files = file_names[i:i + self.batch_size]
            print(f"Downloading batch: {batch_files}")
            for file_name in batch_files:
                print(f"Downloading file: {file_name}")
                self.api.dataset_download_file(self.dataset_name, file_name, path=self.dataset_dir)

                # Check if the file is a zip file and unzip it
                if file_name.endswith('.zip'):
                    self.unzip_file(file_name)

    def unzip_file(self, file_name):
        """Unzip the downloaded file if it is a zip file."""
        zip_file_path = os.path.join(self.dataset_dir, file_name)

        # Create a directory to extract the files
        extract_dir = os.path.join(self.dataset_dir, os.path.splitext(file_name)[0])
        os.makedirs(extract_dir, exist_ok=True)

        # Unzip the file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Unzipped: {zip_file_path} to {extract_dir}")

    def get_dataset_dir(self):
        """Return the dataset directory path."""
        return self.dataset_dir


#Usage example
# downloader = Dataloader('C:/Test', 'benjaminwarner/resized-2015-2019-blindness-detection-images',
#                                   batch_size=2)
# downloader.download_dataset_in_batches()


#
# import subprocess
# import sys
# import os
# import platform
#
# from kaggle import api
#
# from my_code.Util.MultiPlatform import standardize_path
#
#
# def update_Global_os_param():
#     """# Add the project root directory to the Python path """
#     sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
#     print(f"{platform.system()} platform detected. - Global OS parameters updated")
#
#
# update_Global_os_param()# ref to all the Import from internal folders
#
# if platform.system() == "Windows":
#     # load progress bar only for windows - currently
#     from tqdm import tqdm
#     print("Windows platform detected. -Global OS parameters updated")
#
# from Util.MultiPlatform import *
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from kaggle.api.kaggle_api_extended import KaggleApi
#
# class KaggleDataDownLoader:
#     def __init__(self, dataset_path , dataset_name, kaggle_json_path = None, batch_size=50 ):
#         self.dataset_name = dataset_name
#         self.dataset_path = dataset_path
#         self.dataset_dir = os.path.join(self.dataset_path, self.dataset_name)
#         self.kaggle_json_path = kaggle_json_path
#         self.batch_size = batch_size
#         print(f"dataset_path {self.dataset_path} dataset {self.dataset_name} , kaggle_json_path = {self.kaggle_json_path}")
#         self.create_kaggle_json()
#         self.setup_kaggle_api()
#
#     def create_kaggle_json(self):
#         """
#         Create the Kaggle API JSON file if it doesn't exist.
#         """
#         if self.kaggle_json_path is None:
#             # Find the pre-installed kaggle API key          #home_directory = get_home_directory()
#             # Get the home directory dynamically
#             home_directory = os.path.expanduser("~")
#             # Fixing the construct_path call to handle the home_directory correctly
#             self.kaggle_json_path = standardize_path(home_directory, '.kaggle', 'kaggle.json')
#             #
#
#             print(self.kaggle_json_path)  # This will print the standardized path to the kaggle.json file # remove
#             # self.kaggle_json_path = os.path.join(home_directory, '.kaggle', 'kaggle.json') #  remove
#             print(self.kaggle_json_path)
#
#         if not os.path.exists(self.kaggle_json_path):
#             raise FileNotFoundError(f"Kaggle JSON file not found at {self.kaggle_json_path}")
#
#     def setup_kaggle_api(self):
#         """
#         Setup Kaggle API using the JSON file.
#         """
#         try:
#
#             self.dataset_dir = os.path.join(self.dataset_path, self.dataset_name)
#             if os.path.isdir(self.dataset_dir):  # Check if the dataset is already downloaded
#                 print(f"Dataset {self.dataset_name} was already downloaded.")
#                 return
#             else:
#                 print(f"Dataset {self.dataset_dir} not found - downloading dataset {self.dataset_name}")
#                 # Directly download dataset using subprocess and the working Kaggle command
#                 #GY select one took other option.
#                 api.dataset_download_files(self.dataset_name, path=self.dataset_dir, unzip=True)
#                 # subprocess.run(['kaggle', 'datasets', 'download', '-d', self.dataset_name, '-p',
#                 # self.dataset_dir],check=True)
#                 # Extract dataset files if needed (assuming it's a zip file)
#                 zip_file_path = os.path.join(self.dataset_dir, f'{self.dataset_name}.zip')
#                 if os.path.exists(zip_file_path):
#                     print(f"Extracting {zip_file_path} ...")
#                     subprocess.run(['unzip', zip_file_path, '-d', self.dataset_dir], check=True)
#                     print("Extraction complete.")
#
#         except subprocess.CalledProcessError as e:
#             print(f"Error during dataset download or extraction: {e}")
#         except Exception as e:
#             print(f"An error occurred: {e}")
#
#     def get_dataset_dir(self):
#         return self.dataset_dir

#     # The data is already split into train and test and validation;
#     # Data loader implemented in dataPreparation.py
#
# #===============================================================================
# # Download datasets into DATASET_PATH
# # kaggle_loader = KaggleDataDownLoader(DATASET_NAME)
# # kaggle_loader.load_data(os.path.join(DATASET_PATH, 'labels'), 'trainLabels15.csv')
# # kaggle_loader.preprocess_data()
# # kaggle_loader.split_data()
# # train_data = kaggle_loader.get_train_data()
# # validation_data = kaggle_loader.get_validation_data()
# # test_data = kaggle_loader.get_test_data()
