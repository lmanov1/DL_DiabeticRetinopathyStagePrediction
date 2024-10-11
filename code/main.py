# This will be run manually by the user to check for GPU availability and install CUDA if necessary
# from Util import check_hardware_and_install as check_hardware_and_install
# check_hardware_and_install.install_dependencies()

# Import necessary modules
import torch
import os
# Import your model class# Import your model class
from models import inference_model as Neural_Net
from data import data_preparation as data_preparation
from data import data_preprocessing as data_preprocessing
from data import Dataloader as DataLoader


# Define the global variable for the dataset path
DATASET_PATH = '/path/to/your/dataset'

# Choose here the dataset(s) you want to download (https://www.kaggle.com/search?q=APTOS+2019+Blindness+Detection+Dataset+in%3Adatasets)
# Dataset names can be found under the  https://www.kaggle.com/datasets page
DATASET_NAME_resized15_19 = 'benjaminwarner/resized-2015-2019-blindness-detection-images'   # 18.75 GB  
DATASET_NAME_aptos19 = 'mariaherrerot/aptos2019'   # 8.6GB
DATASET_PATH = 'data/raw/'

# Define the dataset structure
dataset_train_structure_resized15_19 = [
    {
        'labels': 'labels/trainLabels15.csv',
        'images': 'resized train 15'
    },
    {
        'labels': 'labels/trainLabels19.csv',
        'images': 'resized train 19'
    }
]

DATASETS = [ DATASET_NAME_resized15_19 , DATASET_NAME_aptos19] 

# Define your main function
def main():

    # data = None
    # model = None
    # # Example data transfer
    print("Starting the main function...")
    # Download datasets into DATASET_PATH
    print("Downloading datasets...")
    kaggle_loader = DataLoader.KaggleDataLoader(DATASET_PATH , DATASETS[0])
    print(f"Dataset dawnloaded into {kaggle_loader.dataset_dir}")
    print("Loading datasets...")
    labels15 = kaggle_loader.load_data( dataset_train_structure_resized15_19[0]['labels'])  
    #images15 = kaggle_loader.load_data( dataset_train_structure_resized15_19[0]['images'])

    print(f"Labels: {labels15}")
    #print(f"Images: {images15}")
    
    # data already split into train and test
    
    # kaggle_loader.preprocess_data()
    # kaggle_loader.split_data()
    # train_data = kaggle_loader.get_train_data()
    # validation_data = kaggle_loader.get_validation_data()
    # test_data = kaggle_loader.get_test_data()

    # Check if CUDA is available and set the device accordingly
    print("Checking for CUDA availability...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # data = data_preparation.DataPreparation(device)
    # model = Neural_Net.to(device)
    # # Your main code logic here
    # print("Starting the main function...")
    # # Example model and data transfer
    # model_inference = Neural_Net.ModelInterface(model)
    # data_prepared = data_preparation.DataPreparation(data)
    #
    #

    # Add your model training/evaluation code
    print("Model and data are set up and ready!")


# Call the main function
if __name__ == "__main__":
    main()
