import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fastai.vision.all import *
from code.Util.MultiPlatform import *

# This class is used to load previously downloaded data from a CSV file (for labels) and a correstponding 
# folder (for images)

# class DataPreparation:
#     def __init__(self, file_path):
#         self.file_path = file_path
#         self.data = None
#         self.train_data = None
#         self.test_data = None
#         self.scaler = StandardScaler()
# def load_data(self):
#     """
#     Load data from a CSV file.
#     """
#     self.data = pd.read_csv(self.file_path)
#     print(f"Data loaded from {self.file_path}")
#     def preprocess_data(self):
#         """
#         Preprocess data, such as scaling and handling missing values.
#         """
#         # Handle missing values
#         self.data = self.data.dropna()

#         # Scaling features
#         features = self.data.drop('target', axis=1)
#         target = self.data['target']
#         features_scaled = self.scaler.fit_transform(features)

#         self.data = pd.DataFrame(features_scaled, columns=features.columns)
#         self.data['target'] = target
#         print("Data preprocessed")

#     def split_data(self, test_size=0.2):
#         """
#         Split data into training and testing sets.
#         """
#         features = self.data.drop('target', axis=1)
#         target = self.data['target']
#         self.train_data, self.test_data = train_test_split(self.data, test_size=test_size, random_state=42)
#         print(f"Data split into training and testing sets with test size = {test_size}")

#     def get_train_data(self):
#         return self.train_data

#     def get_test_data(self):
#         return self.test_data

# Example usage
# data_prep = DataPreparation('path_to_your_csv_file.csv')
# data_prep.load_data()
# data_prep.preprocess_data()
# data_prep.split_data()
# train_data = data_prep.get_train_data()
# test_data = data_prep.get_test_data()

# This class is used to prepare image data for training and validation using the fastai library.
# Initialization: Takes paths to the CSV file and image folder, along with other parameters like label column, validation split percentage, batch size, and random seed.
# Load Data: Reads the CSV file and creates a DataBlock for image processing, then generates DataLoaders.
# Normalize Data: Normalizes the data using the statistics of the training set and moves the data to GPU if available.
# Augment Data: Applies data augmentation techniques to the training dataset.
# Get DataLoaders: Provides the DataLoaders for training and validation.
# Show Batch: Displays a batch of images with labels for quick visualization.
class DataPreparation:
    def __init__(self, csv_path, img_folder,
                   valid_pct=0.2, batch_size=32, seed=42):
        self.csv_path = csv_path
        self.img_folder = img_folder + get_path_separator()      
        self.valid_pct = valid_pct
        self.batch_size = batch_size
        self.seed = seed
        self.dls = None

    def load_data(self):
        """
        Load data from a CSV file and create a DataBlock for image processing.
        """
        df = pd.read_csv(self.csv_path)
        print(f"Data loaded from {self.csv_path}")               
        df[df.columns[0]] = df[df.columns[0]].astype(str) + '.jpg'

        print("Checking for missing files...")
        print("df shape Before: ", df.shape)

        # Check if the image files exist    
        missing_files = df[~df[df.columns[0]].apply(lambda x: (self.img_folder + x)).map(Path).map(Path.exists)]
        if not missing_files.empty:
            print(f"Missing files {missing_files.shape}:")
            print(missing_files)
            # Remove rows with missing files
            df = df[df[df.columns[0]].apply(lambda x: (self.img_folder + x)).map(Path).map(Path.exists)]
            #df.dropna(inplace=True)
            print("After removing missing files, df shape: ", df.shape)

        print(" df columns: ", df.columns)
        df.info()        
        df.head()
        df.value_counts()
        try:
            dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_x=ColReader(df.columns[0], pref=self.img_folder),
            get_y=ColReader(df.columns[1]),
            splitter=RandomSplitter(valid_pct=self.valid_pct, seed=self.seed),
            item_tfms=Resize(224), # Updated it was 460
            batch_tfms=aug_transforms(size=224, min_scale=0.75)
            )    
        except FileNotFoundError:
            print(f"File not found. Skipping to the next file.")
            
            pass
        
        self.dls = dblock.dataloaders(df, bs=self.batch_size)
        print(f"data loaders : {self.dls}")

    def normalize_data(self):
        """
        Normalize the data using the statistics of the training set.
        """
        if self.dls is None:
            self.load_data()

        if torch.cuda.is_available():
            self.dls = self.dls.cuda()
        else:
            print("CUDA not available. Moving data to CPU.")

        self.dls.normalize()
        print("Data normalized")

    def augment_data(self):
        """
        Apply data augmentation techniques.
        """
        if self.dls is None:
            self.load_data()

        self.dls.train.dataset = self.dls.train.dataset.new(item_tfms=aug_transforms(mult=2))
        print("Data augmentation applied")

    def get_dataloaders(self):
        """
        Get the DataLoaders for training and validation.
        """
        if self.dls is None:
            self.load_data()

        return self.dls

    def show_batch(self, n=9):
        """
        Show a batch of images with labels.
        """
        if self.dls is None:
            self.load_data()

        self.dls.show_batch(n=n)

# Example usage
# if __name__ == "__main__":
#     csv_path = 'path/to/your/labels.csv'
#     img_folder = 'path/to/your/images'
#     data_prep = DataPreparation(csv_path, img_folder)
#     data_prep.load_data()
#     data_prep.normalize_data()
#     data_prep.augment_data()
#     dls = data_prep.get_dataloaders()
#     data_prep.show_batch()