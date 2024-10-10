import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreparation:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.train_data = None
        self.test_data = None
        self.scaler = StandardScaler()

    def load_data(self):
        """
        Load data from a CSV file.
        """
        self.data = pd.read_csv(self.file_path)
        print(f"Data loaded from {self.file_path}")

    def preprocess_data(self):
        """
        Preprocess data, such as scaling and handling missing values.
        """
        # Handle missing values
        self.data = self.data.dropna()

        # Scaling features
        features = self.data.drop('target', axis=1)
        target = self.data['target']
        features_scaled = self.scaler.fit_transform(features)

        self.data = pd.DataFrame(features_scaled, columns=features.columns)
        self.data['target'] = target
        print("Data preprocessed")

    def split_data(self, test_size=0.2):
        """
        Split data into training and testing sets.
        """
        features = self.data.drop('target', axis=1)
        target = self.data['target']
        self.train_data, self.test_data = train_test_split(self.data, test_size=test_size, random_state=42)
        print(f"Data split into training and testing sets with test size = {test_size}")

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

# Example usage
# data_prep = DataPreparation('path_to_your_csv_file.csv')
# data_prep.load_data()
# data_prep.preprocess_data()
# data_prep.split_data()
# train_data = data_prep.get_train_data()
# test_data = data_prep.get_test_data()
