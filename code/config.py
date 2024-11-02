
import torch
current_model_name = "resnet50"
#current_model_name = "resnet152" 
#current_model_name = "vgg16_bn"
#current_model_name = "efficientnet-b7"
current_model_name = current_model_name.lower()
split_data_dataloaders = True
testset_evaluation_if_exists = False
handle_data_disbalance = False
tune_find_lr = False

# Check if CUDA is available and set the device accordingly
print("Checking for CUDA availability...") 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the dataset names and paths
# # Choose here the dataset(s) you want to download (https://www.kaggle.com/search?q=APTOS+2019+Blindness+Detection+Dataset+in%3AALL_DATASETS)
# # Dataset names can be found under the  https://www.kaggle.com/ALL_DATASETS page
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

# testLabels15 is bigger dataset then trainLabels15 
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
        "test": dataset_test_structure_resized15_19, # controlled by testset_evaluation_if_exists; if False, this is None
        "val": None,
        "imaging_format": ".jpg",
        "name": "resized15_19",
    },    
    DATASET_NAME_aptos19: {
        "train": dataset_train_structure_aptos19,
        "test": dataset_test_structure_aptos19, # controlled by testset_evaluation_if_exists; if False, this is None
        "val": dataset_val_structure_aptos19,
        "imaging_format": ".png",
        "name": "aptos19",
    },
    # Add other datasets and their structures here if needed
}