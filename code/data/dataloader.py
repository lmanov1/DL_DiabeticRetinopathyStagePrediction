

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from kaggle.api.kaggle_api_extended import KaggleApi
from Util.MultiPlatform import *
#import json


class KaggleDataDownLoader:
    def __init__(self, dataset_path , dataset_name, kaggle_json_path = None):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.kaggle_json_path = kaggle_json_path
        print(f"dataset_path {self.dataset_path} dataset {self.dataset_name} , kaggle_json_path = {self.kaggle_json_path}")
        self.create_kaggle_json()
        self.setup_kaggle_api()

    def create_kaggle_json(self):
        """
        Create the Kaggle API JSON file if it doesn't exist.
        """
        if (self.kaggle_json_path == None):
            # Find the pre-installed kaggle API key
            home_directory = get_home_directory()
            self.kaggle_json_path = os.path.join(home_directory, '.kaggle', 'kaggle.json')
            print(self.kaggle_json_path)
        
        if not os.path.exists(self.kaggle_json_path):
            raise FileNotFoundError(f"Kaggle JSON file not found at {self.kaggle_json_path}")

    def setup_kaggle_api(self):
        """
        Setup Kaggle API using the JSON file.
        """
        os.environ['KAGGLE_CONFIG_DIR'] = os.path.dirname(self.kaggle_json_path)
        api = KaggleApi()
        api.authenticate()
        print("Kaggle API setup complete")
        self.dataset_dir = os.path.join(self.dataset_path, self.dataset_name )
        self.dataset_dir +=  get_path_separator()
        #print(f"Dataset directory: {self.dataset_dir}")
        if os.path.isdir(self.dataset_dir):
            print(f"Dataset {self.dataset_name} was already downloaded =========")
            return 
        else:
            print(f"Dataset {self.dataset_dir} not found - downloading dataset {self.dataset_name}" )     
        
        api.dataset_download_files(self.dataset_name, path=self.dataset_dir, unzip=True)
        print(f"Dataset {self.dataset_name} downloaded")
        return 
    
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
