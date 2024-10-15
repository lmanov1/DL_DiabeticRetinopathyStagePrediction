# import os
# import zipfile
# from kaggle.api.kaggle_api_extended import KaggleApi
#
#
# class DatasetDownloader:
#     def __init__(self, dataset_name, batch_size=5):
#         self.api = KaggleApi()
#         self.api.authenticate()  # Ensure you are authenticated
#         self.dataset_name = dataset_name
#         self.batch_size = batch_size
#         self.download_path = r'C:\Test'  # Set the path to save downloads
#
#         # Create the directory if it doesn't exist
#         os.makedirs(self.download_path, exist_ok=True)
#
#     def download_dataset_in_batches(self):
#         # Get the list of files in the dataset
#         dataset_files = self.api.dataset_list_files(self.dataset_name)
#
#         # Extract file names from ListFilesResult
#         file_names = [file.name for file in dataset_files.files]  # Access the name attribute directly
#
#         # Download files in batches
#         for i in range(0, len(file_names), self.batch_size):
#             batch_files = file_names[i:i + self.batch_size]
#             print(f"Downloading batch: {batch_files}")
#             for file_name in batch_files:
#                 print(f"Downloading file: {file_name}")
#                 self.api.dataset_download_file(self.dataset_name, file_name, path=self.download_path)
#
#                 # Check if the file is a zip file and unzip it
#                 if file_name.endswith('.zip'):
#                     self.unzip_file(file_name)
#                 else:
#                     print(f"File downloaded: {file_name}")
#
#     def unzip_file(self, file_name):
#         # Define the file path
#         zip_file_path = os.path.join(self.download_path, file_name)
#
#         # Create a directory to extract the files
#         extract_dir = os.path.join(self.download_path, os.path.splitext(file_name)[0])
#         os.makedirs(extract_dir, exist_ok=True)
#
#         # Unzip the file
#         with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#             zip_ref.extractall(extract_dir)
#         print(f"Unzipped: {zip_file_path} to {extract_dir}")
#
#
# # Usage example
# downloader = DatasetDownloader('benjaminwarner/resized-2015-2019-blindness-detection-images', batch_size=2)
# downloader.download_dataset_in_batches()

# exit()


import sys
import os
import platform

from data.dataloader import Dataloader


def update_Global_os_param():
    """# Add the project root directory to the Python path """
    if platform.system() == "Windows":
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
        print("Windows platform detected. -Global OS parameters updated")


update_Global_os_param() # ref to all the Import from internal folders
import subprocess

import sys
# Add the project root directory to the Python path

from torchvision.models import vgg16, VGG16_Weights
from Util.MultiPlatform import *
from models.train_model import EyeDiseaseClassifier, PretrainedEyeDiseaseClassifier, pretrained_models
from data.dataPreparation import DataPreparation
from data.dataloader import Dataloader

import torch
import torch.nn as nn
import torch.optim as optim
from fastai.vision.all import *
import time

# Example usage
DATASET_PATH = ('data/raw')
DATASET_NAME_resized15_19 = ('benjaminwarner/resized-2015-2019-blindness-detection-images')
DATASET_NAME_aptos19 = ('mariaherrerot/aptos2019')

# Define the dataset structure for validation and test datasets
dataset_train_structure_resized15_19 = [
    {
        'labels': (DATASET_NAME_resized15_19, 'labels', 'trainLabels15.csv'),
        'images': (DATASET_NAME_resized15_19, 'resized train 15')
    },
    {
        'labels': (DATASET_NAME_resized15_19, 'labels', 'trainLabels19.csv'),
        'images': (DATASET_NAME_resized15_19, 'resized train 19')
    }
]

# DATASET_PATH = standardize_path('data/raw')
# DATASET_NAME_resized15_19 = standardize_path('benjaminwarner/resized-2015-2019-blindness-detection-images')
# DATASET_NAME_aptos19 = standardize_path('mariaherrerot/aptos2019')
#
# dataset_train_structure_resized15_19 = [
#     {
#         'labels': construct_path(DATASET_NAME_resized15_19, 'labels', 'trainLabels15.csv'),
#         'images': construct_path(DATASET_NAME_resized15_19, 'resized train 15')
#     },
#     {
#         'labels': construct_path(DATASET_NAME_resized15_19, 'labels', 'trainLabels19.csv'),
#         'images': construct_path(DATASET_NAME_resized15_19, 'resized train 19')
#     }
# ]

print(DATASET_PATH)
print(DATASET_NAME_resized15_19)
print(DATASET_NAME_aptos19)
print("Dataset Train Structure Resized 15-19:", dataset_train_structure_resized15_19)

DATASETS = [DATASET_NAME_resized15_19, DATASET_NAME_aptos19]


# Check for NVIDIA GPU and install the correct TensorFlow version
def is_nvidia_gpu_available():
    nvidia_smi_command = "nvidia-smi --query-gpu count --format=csv"
    nvidia_smi_command_list = nvidia_smi_command.split()
    try:
        if platform.system() not in ["Linux", "Windows", "Darwin"]:
            print("Unsupported platform")
            return False
        print("Checking for NVIDIA GPU...")
        print("Running command: ", nvidia_smi_command)
        print("Platform: ", platform.system())

        if platform.system() == "Darwin":  # macOS
            result = subprocess.run(["/usr/local/cuda/bin/nvcc", "--version"], check=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
        else:  # Linux and Windows
            result = subprocess.run(nvidia_smi_command_list, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        gpu_count = int(result.stdout.decode().split('\n')[1].strip())
        print(f"Number of NVIDIA GPUs available: {gpu_count}")
        return gpu_count > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_tensorflow():
    cpu_brand = platform.processor()
    if is_nvidia_gpu_available():
        print("Nvidia GPU found. Installing CUDA-supported TensorFlow...")
        if platform.system() == "Windows":
            os.system("winget install cuda")  # Assuming you use winget, adjust command as necessary
        elif platform.system() == "Darwin":  # macOS
            os.system("brew install cuda")
        else:  # Linux
            os.system("sudo apt-get install cuda")

        tensorflow_version = "tensorflow-gpu==2.17.0"  # Or another appropriate version for Nvidia
        io_version = "tensorflow-io-gcs-filesystem==0.37.1"
    else:
        print("No GPU detected. Installing CPU-only TensorFlow...")
        tensorflow_version = "tensorflow==2.12.0"
        io_version = "tensorflow-io-gcs-filesystem==0.29.0"

    subprocess.check_call([sys.executable, '-m', 'pip', 'install', tensorflow_version])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', io_version])

def dataset_exists(dataset, dataset_type: int, dataset_key: str) -> bool:
    """
    Check if the label file and image directory exist for the given dataset.

    Parameters:
    - dataset (dict): A dictionary containing dataset information.
    - dataset_type (int): The type of dataset to check (15 or 19).
    - dataset_key (str): The key to determine the specific dataset.

    Returns:
    - bool: True if both the label file and image directory exist, False otherwise.
    """
    # Check if 'labels' and 'images' are in the dataset dictionary
    if 'labels' in dataset and 'images' in dataset:
        labels = dataset['labels']
        images = dataset['images']

        # Ensure the tuple has the expected number of elements
        if len(labels) == 3 and len(images) == 2:
            # Extract label file components
            base_path = labels[0]  # e.g., 'benjaminwarner/resized-2015-2019-blindness-detection-images'
            label_directory = labels[1]  # e.g., 'labels'
            label_file = labels[2]  # e.g., 'trainLabels15.csv'

            # Construct the full path to the label file
            full_label_path = os.path.join(DATASET_PATH, base_path, label_directory, label_file)

            # Check if the label file exists
            if not os.path.exists(full_label_path):
                print(f"Label file does not exist: {full_label_path}")
                return False  # Return False if label file doesn't exist

            # Check for images based on the dataset type (15 or 19)
            if dataset_type in [15, 19]:  # Ensure type is valid
                image_directory = images[1]  # e.g., 'resized train 15' or 'resized train 19'
                full_image_path = os.path.join(DATASET_PATH, base_path, image_directory)

                # Check if the image directory exists
                if os.path.exists(full_image_path):
                    print(f"Image directory exists: {full_image_path}")
                    return True  # Return True if both label file and image directory exist
                else:
                    print(f"Image directory does not exist: {full_image_path}")
                    return False  # Return False if image directory doesn't exist

    return False  # Return False if the structure is not valid


def main():

    install_tensorflow()  # Install TensorFlow dynamically based on hardware
    train_dataloaders = {}
    print("Starting the main function...")

    print("Downloading datasets...")
    #kaggle_loader = Dataloader(DATASET_PATH, DATASETS[0])

    # # Set Kaggle credentials
    # kaggle_loader.set_credentials('your_kaggle_username', 'your_kaggle_key')

    #print(f"Dataset will be downloaded into {kaggle_loader.dataset_path}")
    print("Loading train aptos 19 dataset...")

    # Proceed to download
    # The download will happen during the initialization process

    print("Downloading datasets...")

    # Initialize the KaggleDataDownLoader with the appropriate parameters

    # Print the parameters
    print(f"Dataset Path: {DATASET_PATH}")
    print(f"Dataset Name: {DATASETS[0]}")


    # gpt PRINT THE PARAMS BEFORE INITIATE THE Dataloader
    kaggle_loader = Dataloader(
        dataset_path=DATASET_PATH,
        dataset_name=DATASETS[0],
        batch_size=5  # Adjust this batch size as needed
    )
    #Dataloader('C:/Test', 'benjaminwarner/resized-2015-2019-blindness-detection-images',
               #                                   batch_size=2)


    # Inform the user where the dataset has been downloaded
    print(f"Dataset downloaded into {kaggle_loader.get_dataset_dir()}")

    print("Loading train aptos 19 dataset...")

    # Iterate through the dataset structure
    for dataset in dataset_train_structure_resized15_19:
        print("---------------------------------------")
        print(dataset)

        # Construct paths for labels and images using the new dataset directory
        dataset['labels'] = (kaggle_loader.get_dataset_dir().strip('/'), dataset['labels'])
        dataset['images'] = (kaggle_loader.get_dataset_dir(), dataset['images'])

        # dataset['labels'] = construct_path(kaggle_loader.get_dataset_dir().strip('/'), dataset['labels'])
        # dataset['images'] = construct_path(kaggle_loader.get_dataset_dir(), dataset['images'])

        print(f"{dataset['labels']} \n {dataset['images']}")

        # Check if the label file exists
        # Example of using the function

        # Assuming you have already defined the dataset_exists function and DataPreparation class
        dataloader = None

        for dataset1 in dataset_train_structure_resized15_19:
            # Check that the labels tuple has at least 3 elements
            if len(dataset1['labels']) >= 3:
                dataset_type = 15 if '15' in dataset1['labels'][2] else 19  # Determine dataset type from label file
            else:
                print(f"Skipping dataset due to invalid labels structure: {dataset1['labels']}")
                continue

            # Check if the dataset exists
            exists = dataset_exists(dataset1, dataset_type, 'resized15_19')  # You can use dataset_key if needed
            print(f"Dataset existence result: {exists}")

            # If the dataset does not exist, skip to the next one
            if not exists:
                print("Continuing to the next dataset...")
                continue

            # Ensure that images tuple has at least 2 elements before accessing
            if len(dataset1['images']) < 2:
                print(f"Skipping dataset due to invalid images structure: {dataset1['images']}")
                continue

            # example of the path to: my_code\data\raw\benjaminwarner\resized-2015-2019-blindness-detection-images
            # there is csv trainLabels15 for train then there are images to analyze


            # Extract paths for loading data
            img_folder_path = os.path.join(DATASET_PATH, dataset1['images'][0],
                                           dataset1['images'][1])  # Correctly join the image folder path
            csv_path = os.path.join(DATASET_PATH, dataset1['labels'][0], dataset1['labels'][1], dataset1['labels'][2])

            # Debugging output to check paths
            print(f"Image folder path: {img_folder_path}")
            print(f"CSV path: {csv_path}")

            # @GY why
            #Exanine why it was mixed \/ :Mixing forward and backward slashes can indeed cause file path issues.Here’s how to keep it
            #consistent: Replace backslashes with forward slashes or use Python’s os.path.join() method for cross-platform compatibility.


            # Create an instance of DataPreparation
            dataloader = DataPreparation(csv_path=csv_path, img_folder=img_folder_path)

            # Load data using the dataloader
            dataloader.load_data()

            # Get the data loaders for training
            dls = dataloader.get_dataloaders()

            # Optionally display a batch of the data
            dls.show_batch()

            # Store the data loaders in a dictionary with the labels path as the key
            train_dataloaders[dataset1['labels']] = dls
            print("=====================================")

            # dataloader.normalize_data()
            # dataloader.augment_data()



    print(train_dataloaders)
    # Check if CUDA is available and set the device accordingly
    print("Checking for CUDA availability...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_dr_classes = 5
    # 0 - No DR
    # 1 - Mild
    # 2 - Moderate
    # 3 - Severe
    # 4 - Proliferative DR

    pretrained_model = PretrainedEyeDiseaseClassifier(num_classes=num_dr_classes, pretrained_model=pretrained_models[0])
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLossFlat
    # pretr_optimizer = optim.Adam(pretrained_model.parameters(), lr=0.001)

    inf_model = EyeDiseaseClassifier(num_classes=num_dr_classes)
    # inf_optimizer = optim.Adam(inf_model.parameters(), lr=0.001)

    inf_model.to(device)
    pretrained_model.to(device)

    for key, dls in train_dataloaders.items():
        # 1. Train the (pretrained) model
        print("Current directory:", os.getcwd())
        # Extract the key from the path
        dataset_name = os.path.basename(key).split('.')[0]
        pretrained_model_file_name = dataset_name + '_pretrained_model.pth'
        pretrained_weigths_path = os.path.join(os.getcwd(), 'data', 'output', pretrained_model_file_name).replace('/', get_path_separator())
        print(" \n ===>  Looking for pretrained model here", pretrained_weigths_path)      # 513 MB
        # very heavy run - about 8 hours on 100% GPU - lets not run it again
        if not os.path.exists(pretrained_weigths_path): 
            print("Pretrained model not found - training now...")
            start_time = time.time()
            pretrained_model.train_model(dls, epochs=10)
            end_time = time.time()
            print(f"==> Pretrained model training time: {end_time - start_time} seconds")
            # Save the pretrained model weights
            torch.save(pretrained_model.state_dict(), pretrained_weigths_path)
            print(" ===>  Saving pretrained model to ", pretrained_weigths_path)
            # Evaluate the model
            pretrained_model.evaluate_model(dls)
        else:
            print(" ===>  Pretrained model already exists at ", pretrained_weigths_path)
            print("Going to train CNN model...")

        # 2. Train model
        # Load the pretrained weights into the main model
        # inf_model.load_state_dict(torch.load(pretrained_weigths_path)) - doesn't work need to see how to load the weights        
        inf_learner = inf_model.get_learner(dls, criterion, accuracy)
        #Training (fit_one_cycle): We use fit_one_cycle for training the model from scratch, as it’s more suited 
        # for models without pretrained weights.
        start_time = time.time()
        inf_learner.fit_one_cycle(10)
        end_time = time.time()
        print_time(start_time , end_time , "CNN model training time")
        # Save the trained model weights
        trained_model_file_name = dataset_name + '_trained_model.pth'
        trained_weigths_path = os.path.join(os.getcwd(), 'data', 'output', trained_model_file_name).replace('/', get_path_separator())        
        print(" ===> Saving train model to ", trained_weigths_path)
        torch.save(inf_model.state_dict(), trained_weigths_path )
        inf_model.evaluate_model(dls)    

# Call the main function
if __name__ == "__main__":
    main()
