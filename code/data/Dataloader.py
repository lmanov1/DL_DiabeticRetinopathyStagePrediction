

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from kaggle.api.kaggle_api_extended import KaggleApi
from code.Util.MultiPlatform import *

# Define the dataset names and paths
# Choose here the dataset(s) you want to download (https://www.kaggle.com/search?q=APTOS+2019+Blindness+Detection+Dataset+in%3AALL_DATASETS)
# Dataset names can be found under the  https://www.kaggle.com/ALL_DATASETS page
DATASET_NAME_resized15_19 = (
    "benjaminwarner/resized-2015-2019-blindness-detection-images"  # 18.75 GB
)
DATASET_NAME_aptos19 = "mariaherrerot/aptos2019"  # 8.6GB
DATASET_PATH = "data/raw/"

# Define the ALL_DATASETS structure .. here are tests and train ALL_DATASETS

dataset_train_structure_resized15_19 = [
    {
        "labels": "labels/trainLabels15.csv",
        "images": "resized train 15",  # 6.66GB on disk , 35126 files
    }
    # ,{
    #     'labels': 'labels/trainLabels19.csv',
    #     'images': 'resized train 19'  # 630MB on disk , 3662 files
    # }
]

# testLabels15 is bigger dataset then trainLabels15 , so will use it for training and vise versa
#

dataset_test_structure_resized15_19 = [
    {
        "labels": "labels/testLabels15.csv",  # 53577
        "images": "resized test 15",  # 11GB on disk , 53577 files
    }
    #,{
    #     'labels': 'labels/testImages19.csv',  # no labels available for test19 so this dataset is unusable 
    #     for evaluation/hyperparameters tuning but valid for inference which is less interesting
    #     'images': 'resized train 19'  # 630MB on disk , 3662 files
    # }
]

dataset_train_structure_aptos19 = [{
        "labels": "train_1.csv",  # 2930 rows
        "images": "train_images/train_images",  # 6.4 GB 2930 images
    }]
dataset_test_structure_aptos19 = [{
        "labels": "test.csv",  #  need to crop as file has 366 rows with data and all the rest is empty
        "images": "test_images/test_images",  # 796 MB 366 images
    }]
dataset_val_structure_aptos19 = [{
        "labels": "valid.csv",  #  366 rows
        "images": "val_images/val_images",  # 902 MB   366 images
    }]

DATASET_NAMES = [DATASET_NAME_resized15_19, DATASET_NAME_aptos19]

# Create a data structure to map each dataset to its associated train and test structures (and val if exists)
ALL_DATASETS = {
    DATASET_NAME_resized15_19: {
        "train": dataset_train_structure_resized15_19,
        "test": dataset_test_structure_resized15_19,
        "val": None,
        "imaging_format": ".jpg",
        "name": "resized15_19",
    },    
    DATASET_NAME_aptos19: {
        "train": dataset_train_structure_aptos19,
        "test": dataset_test_structure_aptos19,
        "val": dataset_val_structure_aptos19,
        "imaging_format": ".png",
        "name": "aptos19",
    },
    # Add other datasets and their structures here if needed
}

#===============================================================================

class KaggleDataDownLoader:
    def __init__(self, dataset_path , dataset_name, kaggle_json_path = None):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.kaggle_json_path = kaggle_json_path
        print(f"dataset_path {self.dataset_path} dataset {self.dataset_name} , kaggle_json_path = {self.kaggle_json_path}")        
        self.already_downloaded = False
        self.already_downloaded = self.construct_dataset_path()
        if self.already_downloaded == False:
            print("Downloading dataset from Kaggle")
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
        print(f"Dataset directory: {self.dataset_dir}")
        if os.path.isdir(self.dataset_dir):
            print(f"Dataset {self.dataset_name} was already downloaded =========")
            self.already_downloaded = True
            return True
        return False
    
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


def download_from_kaggle(dataset_name, dataset_path, kaggle_json_path=None):
    """
    Downloads a dataset from Kaggle and saves it to the specified path.
    Args:
        dataset_name (str): The name of the dataset to download from Kaggle.
        dataset_path (str): The local path where the dataset should be saved.
        kaggle_json_path (str, optional): The path to the Kaggle JSON file for authentication. Defaults to None.
    Returns:
        str: The directory where the dataset has been downloaded.
    """
    print(f"Downloading dataset {dataset_name} into {dataset_path}")
    kaggle_downloader = KaggleDataDownLoader(
        dataset_path, dataset_name, kaggle_json_path
    )
    return kaggle_downloader.dataset_dir

#===============================================================================
# Download ALL_DATASETS into DATASET_PATH
# kaggle_loader = KaggleDataDownLoader(DATASET_NAME , DATASET_PATH)
