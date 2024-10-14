import subprocess
import sys
import os
import platform

def update_Global_os_param():
    """# Add the project root directory to the Python path """
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    print(f"{platform.system()} platform detected. - Global OS parameters updated")


update_Global_os_param()# ref to all the Import from internal folders

if platform.system() == "Windows":
    # load progress bar only for windows - currently
    from tqdm import tqdm
    print("Windows platform detected. -Global OS parameters updated")

from Util.MultiPlatform import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from kaggle.api.kaggle_api_extended import KaggleApi

class KaggleDataDownLoader:
    def __init__(self, dataset_path , dataset_name, kaggle_json_path = None, batch_size=50 ):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.dataset_dir = os.path.join(self.dataset_path, self.dataset_name)
        self.kaggle_json_path = kaggle_json_path
        self.batch_size = batch_size
        print(f"dataset_path {self.dataset_path} dataset {self.dataset_name} , kaggle_json_path = {self.kaggle_json_path}")
        self.create_kaggle_json()
        self.setup_kaggle_api()

    def create_kaggle_json(self):
        """
        Create the Kaggle API JSON file if it doesn't exist.
        """
        if ( self.kaggle_json_path == None):
            # Find the pre-installed kaggle API key          #home_directory = get_home_directory()
            # Get the home directory dynamically
            home_directory = os.path.expanduser("~")
            # Fixing the construct_path call to handle the home_directory correctly
            self.kaggle_json_path = standardize_path(home_directory, '.kaggle', 'kaggle.json')
            #

            print(self.kaggle_json_path)  # This will print the standardized path to the kaggle.json file # remove
            #self.kaggle_json_path = os.path.join(home_directory, '.kaggle', 'kaggle.json') #  remove
            print(self.kaggle_json_path)
        
        if not os.path.exists(self.kaggle_json_path):
            raise FileNotFoundError(f"Kaggle JSON file not found at {self.kaggle_json_path}")

        # if not os.path.exists(self.kaggle_json_path):
        #     api_token = {
        #         "username": "your_kaggle_username",
        #         "key": "your_kaggle_key"
        #     }
        #     os.makedirs(os.path.dirname(self.kaggle_json_path), exist_ok=True)
        #     with open(self.kaggle_json_path, 'w') as file:
        #         json.dump(api_token, file)
        #     print(f"{self.kaggle_json_path} created.")
        # else:
        #     print(f"{self.kaggle_json_path} already exists.")

    def authenticate_kaggle(self):
        try:
            api = KaggleApi()
            api.authenticate()
            print("Kaggle API authentication successful.")
            return api
        except Exception as e:
            print(f"Error during Kaggle API authentication: {e}")

    def setup_kaggle_api(self):
        """
        Setup Kaggle API using the JSON file.
        """
        try:
            # Set Kaggle config environment variable
            #os.environ['KAGGLE_CONFIG_DIR'] = os.path.dirname(self.kaggle_json_path) return it later on
            os.environ['KAGGLE_CONFIG_DIR'] = r'C:\Users\DELL\.kaggle'
            # Ensure that the dataset directory exists
            self.dataset_dir = os.path.join(self.dataset_path, self.dataset_name)
            if os.path.isdir(self.dataset_dir):  # Check if the dataset is already downloaded
                print(f"Dataset {self.dataset_name} was already downloaded.")
                return

            print(f"Dataset {self.dataset_dir} not found - downloading dataset {self.dataset_name}.")

            # Directly download dataset using subprocess and the working Kaggle command
            subprocess.run(['kaggle', 'datasets', 'download', '-d', self.dataset_name, '-p', self.dataset_dir],
                           check=True)

            # Extract dataset files if needed (assuming it's a zip file)
            zip_file_path = os.path.join(self.dataset_dir, f'{self.dataset_name}.zip')
            if os.path.exists(zip_file_path):
                print(f"Extracting {zip_file_path} ...")
                subprocess.run(['unzip', zip_file_path, '-d', self.dataset_dir], check=True)
                print("Extraction complete.")

            print(f"Dataset {self.dataset_name} downloaded successfully.")

        except subprocess.CalledProcessError as e:
            print(f"Error during dataset download or extraction: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def get_dataset_dir(self):
        return self.dataset_dir

    # The data is already split into train and test and validation; 
    # Data loader implemented in data_preparation.py

#===============================================================================
# Download datasets into DATASET_PATH
# kaggle_loader = KaggleDataDownLoader(DATASET_NAME)
# kaggle_loader.load_data(os.path.join(DATASET_PATH, 'labels'), 'trainLabels15.csv')
# kaggle_loader.preprocess_data()
# kaggle_loader.split_data()
# train_data = kaggle_loader.get_train_data()
# validation_data = kaggle_loader.get_validation_data()
# test_data = kaggle_loader.get_test_data()
