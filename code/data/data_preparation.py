import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fastai.vision.all import *
from code.Util.MultiPlatform import *

# This class is used to load previously downloaded data from a CSV file (for labels) and a correstponding
# folder (for images)

# This class is used to prepare image data for training and validation using the fastai library.
# Initialization: Takes paths to the CSV file and image folder, along with other parameters like label column, validation split percentage, batch size, and random seed.
# Load Data: Reads the CSV file and creates a DataBlock for image processing, then generates DataLoaders.
# Normalize Data: Normalizes the data using the statistics of the training set and moves the data to GPU if available.
# Augment Data: Applies data augmentation techniques to the training dataset.
# Get DataLoaders: Provides the DataLoaders for training and validation.
# Show Batch: Displays a batch of images with labels for quick visualization.
class DataPreparation:
    def __init__(self, csv_path, img_folder, imaging_format = 'jpg',
                   valid_pct=0.2, batch_size=32, seed=42):
        self.csv_path = csv_path
        self.img_folder = img_folder + get_path_separator()
        self.valid_pct = valid_pct
        self.batch_size = batch_size
        self.imaging_format = imaging_format
        self.seed = seed
        self.dls = None

    def load_data(self):
        """
        Load data from a CSV file and create a DataBlock for image processing.
        """
        df = pd.read_csv(self.csv_path)
        # Ensure the first column values include either '.jpg' or '.png' suffix (defined per dataset with self.imaging_format)
        # Detect which files in the first column don't end with self.imaging_format and add the suffix if needed
        df[df.columns[0]] = df[df.columns[0]].apply(lambda x: x if x.endswith(self.imaging_format) else x  + self.imaging_format)

        # Check if the image files exist in img_folder and remove rows from df if not present
        df = df[df[df.columns[0]].apply(lambda x: (self.img_folder + x)).map(Path).map(Path.exists)]
        print(" df columns: ", df.columns)
        print(f"info: {df.info()}")
        #print(f"head: {df.head()}")
        #print(f"value counts: {df.value_counts()}")

        try:
            dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_x=ColReader(df.columns[0], pref=self.img_folder),
            get_y=ColReader(df.columns[1]),
            #splitter=RandomSplitter(valid_pct=self.valid_pct, seed=self.seed), #Instead of using RandomSplitter, the training dataset is directly used without splitting
            item_tfms=Resize(460),
            batch_tfms=aug_transforms(size=224, min_scale=0.75)
            )
        except FileNotFoundError:
            print(f"File not found. Skipping to the next file.")
            pass

        print(f"DataBlock created  {dblock}")
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