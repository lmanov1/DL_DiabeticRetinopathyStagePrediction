import os
# import json to access Kaggle API

import json
import


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from kaggle.api.kaggle_api_extended import KaggleApi

class KaggleDataLoader:
    def __init__(self, dataset_name, kaggle_json_path):
        self.dataset_name = dataset_name
        self.kaggle_json_path = kaggle_json_path
        self.data = None
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self.scaler = StandardScaler()

        self.ensure_and_set_kaggle_creds()

    def ensure_and_set_kaggle_creds(self):
        """
        Ensures the kaggle.json file exists and sets Kaggle credentials as environment variables.
        """
        kaggle_dir = os.path.dirname(self.kaggle_json_path)

        # Create the directory if it doesn't exist
        if not os.path.exists(kaggle_dir):
            os.makedirs(kaggle_dir)

        # Check if the kaggle.json file exists
        if not os.path.exists(self.kaggle_json_path):
            # Create the kaggle.json file with default data (Replace with actual API token)
            kaggle_json = {
                "username": "your_kaggle_username",
                "key": "your_kaggle_key"
            }
            with open(self.kaggle_json_path, 'w') as f:
                json.dump(kaggle_json, f)
            print(f"Created kaggle.json at {self.kaggle_json_path}")
        else:
            print(f"kaggle.json already exists at {self.kaggle_json_path}")

        # Load the kaggle.json file and set environment variables
        with open(self.kaggle_json_path, 'r') as f:
            kaggle_creds = json.load(f)

        os.environ['KAGGLE_USERNAME'] = kaggle_creds['username']
        os.environ['KAGGLE_KEY'] = kaggle_creds['key']
        print("Kaggle credentials loaded and environment variables set.")

        self.setup_kaggle_api()

    def setup_kaggle_api(self):
        """
        Setup Kaggle API using the JSON file.
        """
        os.environ['KAGGLE_CONFIG_DIR'] = os.path.dirname(self.kaggle_json_path)
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(self.dataset_name, path='.', unzip=True)
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

# Example usage
kaggle_loader = KaggleDataLoader('benjaminwarner/resized-2015-2019-blindness-detection-images', 'C:/Users/DELL/.kaggle/kaggle.json')
kaggle_loader.load_data('labels', 'trainLabels15.csv')
kaggle_loader.preprocess_data()
kaggle_loader.split_data()
train_data = kaggle_loader.get_train_data()
validation_data = kaggle_loader.get_validation_data()
test_data = kaggle_loader.get_test_data()
