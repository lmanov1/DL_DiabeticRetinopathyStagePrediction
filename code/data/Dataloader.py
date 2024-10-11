import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from kaggle.api.kaggle_api_extended import KaggleApi
import platform
import json

# Choose here the dataset(s) you want to download (https://www.kaggle.com/search?q=APTOS+2019+Blindness+Detection+Dataset+in%3Adatasets)
# Dataset names can be found under the  https://www.kaggle.com/datasets page
DATASET_NAME = 'benjaminwarner/resized-2015-2019-blindness-detection-images'   # 18.75 GB  
DATASET_NAME1 = 'mariaherrerot/aptos2019'   # 8.6GB
DATASET_PATH = 'data/raw/'
class KaggleDataLoader:
    def __init__(self, dataset_name, kaggle_json_path):
        self.dataset_name = dataset_name
        self.kaggle_json_path = kaggle_json_path
        self.data = None
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self.scaler = StandardScaler()

        self.create_kaggle_json()
        self.setup_kaggle_api()

    def create_kaggle_json(self):
        """
        Create the Kaggle API JSON file if it doesn't exist.
        """
        if not os.path.exists(self.kaggle_json_path):
            api_token = {
                "username": "your_kaggle_username",
                "key": "your_kaggle_key"
            }
            os.makedirs(os.path.dirname(self.kaggle_json_path), exist_ok=True)
            with open(self.kaggle_json_path, 'w') as file:
                json.dump(api_token, file)
            print(f"{self.kaggle_json_path} created.")
        else:
            print(f"{self.kaggle_json_path} already exists.")

    def setup_kaggle_api(self):
        """
        Setup Kaggle API using the JSON file.
        """
        os.environ['KAGGLE_CONFIG_DIR'] = os.path.dirname(self.kaggle_json_path)
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(self.dataset_name, path=DATASET_PATH, unzip=True)
        print(f"Dataset {self.dataset_name} downloaded")

    def load_data(self, labels_path, file_name):
        """
        Load data from the specified file within the downloaded dataset.
        """
        self.data = pd.read_csv(os.path.join(labels_path, file_name))
        print(f"Data loaded from {file_name}")

    def preprocess_data(self):
        """
        Preprocess data, such as scaling and handling missing values.
        """
        self.data = self.data.dropna()

        # Assuming 'target' is the label column; adjust if necessary
        features = self.data.drop('target', axis=1)
        target = self.data['target']
        features_scaled = self.scaler.fit_transform(features)

        self.data = pd.DataFrame(features_scaled, columns=features.columns)
        self.data['target'] = target
        print("Data preprocessed")

    def split_data(self, validation_size=0.2, test_size=0.2):
        """
        Split data into training, validation, and testing sets.
        """
        features = self.data.drop('target', axis=1)
        target = self.data['target']

        train_data, temp_data = train_test_split(self.data, test_size=test_size, random_state=42)
        self.train_data, self.validation_data = train_test_split(train_data, test_size=validation_size, random_state=42)
        self.test_data = temp_data
        print(f"Data split into training (80%), validation (16%), and testing (20%) sets")

    def get_train_data(self):
        """
        Get the training data.
        """
        return self.train_data

    def get_validation_data(self):
        """
        Get the validation data.
        """
        return self.validation_data

    def get_test_data(self):
        """
        Get the test data.
        """
        return self.test_data


def get_home_directory():
    system = platform.system()
    if system == 'Linux' or system == 'Darwin':  # Darwin is macOS
        home_dir = os.environ.get('HOME')        
    elif system == 'Windows':
        home_dir = os.environ.get('USERPROFILE')
    else:
        raise EnvironmentError("Unsupported operating system")
    
    print(f"Running on {system}. Home directory: {home_dir}")
    return home_dir


#===============================================================================
# Get the kaggle API key
home_directory = get_home_directory()
kaggle_json_path = os.path.join(home_directory, '.kaggle', 'kaggle.json')
print(kaggle_json_path)
if not os.path.exists(kaggle_json_path):
    raise FileNotFoundError(f"Kaggle JSON file not found at {kaggle_json_path}")

# Download datasets into DATASET_PATH
kaggle_loader = KaggleDataLoader(DATASET_NAME1, kaggle_json_path)
kaggle_loader.load_data(os.path.join(DATASET_PATH, 'labels'), 'trainLabels15.csv')
# kaggle_loader.preprocess_data()
# kaggle_loader.split_data()
# train_data = kaggle_loader.get_train_data()
# validation_data = kaggle_loader.get_validation_data()
# test_data = kaggle_loader.get_test_data()
