import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fastai.vision.all import *

from Util.ImageManipulation import image_manip
from code.Util.MultiPlatform import *
from PIL import Image
import matplotlib.pyplot as plt

#from code.Util.ImageManipulation import image_manip
#from code.data.raw.imageManiTransform import ImageManipTransform


# Define your custom callback to apply multiple transformations
def custom_callback(image: Image):
    # Apply multiple transformations in sequence
    # image = Resize(400)(image)  # Resize to 400x400
    # image = Flip(p=0.5)(image)  # Random horizontal flip with 50% chance
    # image = Rotate(max_deg=15)(image)  # Random rotation with max of 15 degrees
    # image = RandomCrop(224)(image)  # Random crop to 224x224
    image = image_manip(image) -
    print("Pipeline: no custom_callback")
    return image


# This class is used to prepare image data for training and validation using the fastai library.
# Initialization: Takes paths to the CSV file and image folder, along with other parameters like label column, validation split percentage, batch size, and random seed.
# Load Data: Reads the CSV file and creates a DataBlock for image processing, then generates DataLoaders.
# Normalize Data: Normalizes the data using the statistics of the training set and moves the data to GPU if available.
# Augment Data: Applies data augmentation techniques to the training dataset.
# Get DataLoaders: Provides the DataLoaders for training and validation.
# Show Batch: Displays a batch of images with labels for quick visualization.
class DataPreparation:
    def __init__(self, csv_path, img_folder,
                   valid_pct=0.2, batch_size=32, seed=42, sampler=None):
        self.csv_path = csv_path
        self.img_folder = img_folder + get_path_separator()
        self.valid_pct = valid_pct
        self.batch_size = batch_size
        self.seed = seed
        self.sampler = None
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
            print("After removing missing files, df shape: ", df.shape)

        print(" df columns: ", df.columns)
        df.info()
        df.head()

        # Ensure proper dataloader creation
        try:
            # Define the DataBlock with transformations
            dblock = DataBlock(
                blocks=(ImageBlock, CategoryBlock),
                get_x=ColReader(df.columns[0], pref=self.img_folder),
                get_y=ColReader(df.columns[1]),
                splitter=RandomSplitter(valid_pct=self.valid_pct, seed=self.seed),
                item_tfms=[
                    Resize(1024),  # Ensures all images are resized to 1024x1024 before custom callback
                    custom_callback  # Custom callback applied after resizing to consistent dimensions
                ],
                batch_tfms=aug_transforms(size=600, min_scale=0.75)
            )

            plt.ion()  # Turn on interactive mode
            # Create dataloaders
            self.dls = dblock.dataloaders(df, bs=self.batch_size, num_workers=0)  # Create dataloaders

            # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            # Show a batch of training data
            print(f"len of self.dls.train:", len(self.dls.train))  # Check the length of the training data
            # self.dls.show_batch(ax=ax)
            # plt.show()  # Display the plot
            print("DataBlock loaded successfully.")

        except Exception as e:
            print(f"Error loading DataBlock: {e}")

        return self.dls

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