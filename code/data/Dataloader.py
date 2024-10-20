

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from kaggle.api.kaggle_api_extended import KaggleApi
from Util.MultiPlatform import *

class KaggleDataDownLoader:
    def __init__(self, dataset_path , dataset_name, kaggle_json_path = None):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.kaggle_json_path = kaggle_json_path
        print(f"dataset_path {self.dataset_path} dataset {self.dataset_name} , kaggle_json_path = {self.kaggle_json_path}")        
        self.already_downloaded = False
        self.already_downloaded = self.construct_dataset_path()
        if self.already_downloaded == False:
            self.create_kaggle_json()
            self.setup_kaggle_api()

    def construct_dataset_path(self):
        """
        Construct the dataset path.
        """
        self.dataset_dir = None
        if self.dataset_path is None:
            raise ValueError("Invalid argument: dataset_path cannot be None")
                           
        self.dataset_dir = os.path.join(self.dataset_path, self.dataset_name)
        self.dataset_dir += get_path_separator()
        #print(f"Dataset directory: {self.dataset_dir}")
        if os.path.isdir(self.dataset_dir):
            print(f"Dataset {self.dataset_name} was already downloaded =========")
            self.already_downloaded = True
            return
    
    def create_kaggle_json(self):
        """
        Create the Kaggle API JSON file if it doesn't exist.
        """
        if ( self.kaggle_json_path == None):
            # Find the pre-installed kaggle API key
            home_directory = get_home_directory()
            self.kaggle_json_path = os.path.join(home_directory, '.kaggle', 'kaggle.json')
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
        print(f"Downloading dataset {self.dataset_name} into {self.dataset_dir}" )             
        api.dataset_download_files(self.dataset_name, path=self.dataset_dir, unzip=True)
        print(f"Dataset {self.dataset_name} was successfully downloaded")
        return 
    
    def get_dataset_dir(self):
        return self.dataset_dir

    # The data is already split into train and test and validation; 
    # Data loader implemented in data_preparation.py

#===============================================================================
# Download datasets into DATASET_PATH
# kaggle_loader = KaggleDataDownLoader(DATASET_NAME , DATASET_PATH)
