# Windows -environment issues by folder name code which has main.py in it

import sys
import os
# This should be the first code that runs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Now you can safely import your modules from the 'code' directory
# from code import your_module  # This should now work as expected

import torch
import torch.nn as nn
from code.models.callbacks.early_stopping import EarlyStopping
from code.models.train_model import EyeDiseaseClassifier, PretrainedEyeDiseaseClassifier, pretrained_models
from code.data import data_preparation as DataPrep
from code.data import Dataloader as KaggleDataLoader
from fastai.vision.all import *
from Util.MultiPlatform import *
from Util.Terminal_Output import save_terminal_output_to_file as terminal_out
from sklearn.model_selection import GridSearchCV  # Import GridSearchCV
import time

# Define the dataset names and paths
DATASET_NAME_resized15_19 = 'benjaminwarner/resized-2015-2019-blindness-detection-images'  # 18.75 GB
DATASET_NAME_aptos19 = 'mariaherrerot/aptos2019'  # 8.6GB
DATASET_PATH = 'data/raw/'

# Define the dataset structure
dataset_train_structure_resized15_19 = [
    {
        'labels': 'labels/trainLabels19.csv',
        'images': 'resized train 19'
    }
]

DATASETS = [DATASET_NAME_resized15_19, DATASET_NAME_aptos19]


# Function to download datasets
def download_datasets(dataset_path, datasets):
    print("Downloading datasets...")
    kaggle_loader = KaggleDataLoader.KaggleDataDownLoader(dataset_path, datasets[0])
    print(f"Dataset downloaded into {kaggle_loader.dataset_dir}")
    return kaggle_loader


# Function to load and prepare data
def load_and_prepare_data(kaggle_loader, dataset_structure):
    train_dataloaders = {}
    for dataset in dataset_structure:
        print("---------------------------------------")
        print(f"Loading dataset structure: {dataset}")
        dataset['labels'] = os.path.join(kaggle_loader.dataset_dir.strip(''), dataset['labels'])
        dataset['images'] = str(os.path.join(kaggle_loader.dataset_dir, dataset['images']))
        print(f"Labels path: {dataset['labels']} \nImages path: {dataset['images']}")
        print("=====================================")

        dataloader = DataPrep.DataPreparation(dataset['labels'], dataset['images'])
        dataloader.load_data()
        dls = dataloader.get_dataloaders()

        # Add print statements to check the sizes of the datasets
        print(f"Loaded {len(dls.train_ds)} training samples.")
        print(f"Loaded {len(dls.valid_ds)} validation samples.")

        dls.show_batch()
        train_dataloaders[dataset['labels']] = dls
        print("=====================================")

    return train_dataloaders


# Function to check for CUDA availability
def check_cuda_availability():
    print("Checking for CUDA availability...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


# Function to train pretrained model
def train_pretrained_model(pretrained_model, dls, pretrained_weights_path):
    if not os.path.exists(pretrained_weights_path):
        print("Pretrained model not found - training now...")
        directory = os.path.dirname(pretrained_weights_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        start_time = time.time()
        pretrained_model.train_model(dls, epochs=10)
        end_time = time.time()
        print_time(start_time, end_time, "Pretrained model training time")

        torch.save(pretrained_model.state_dict(), pretrained_weights_path)
        print(" ===>  Saving pretrained model to ", pretrained_weights_path)
        terminal_out(pretrained_weights_path)
        print(" ===>  Pretrained model training completed.")
        print(" ===>  Evaluating pretrained model...")
        pretrained_model.evaluate_model(dls)
    else:
        print(" ===>  Pretrained model already exists at ", pretrained_weights_path)

    return pretrained_model


# Function to train inference model with GridSearch for hyperparameter tuning
def train_inference_model(inf_model, dls, criterion, quick_debug, patience=5, max_epoch=100):
    """
     Trains a model using the provided data loaders and criterion, implementing early stopping.

     Parameters:
     - inf_model: The model to train.
     - dls: The DataLoaders containing training and validation data.
     - criterion: The loss function used for training.
     - patience: The number of epochs to wait for improvement before stopping (default is 5).
     - quick_debug: If True, limits epochs and data size for quick debugging.
                    If None, uses the provided max_epoch value.
                    if  False, uses 100 as max_epoch value without modifications.
     - max_epoch: The maximum number of epochs to train for (default is 100).


     Returns:
     - inf_model: The trained model after completing the training process.
     """
    #run on mode print_debug print the mode
    print(f"Run one mode Config_kist or Quick Or Const, {quick_debug} ")

    # Quick debug mode to speed up training and find issues faster
    if quick_debug is None:
        max_epochs = max_epoch
        print(f"max_epoch set as the configuration, {max_epoch}")
    elif quick_debug:
        print("Quick debug mode enabled: Reducing epochs and data size for fast issue detection.")
        # Reduce data size by sampling small batches from training/validation sets
        dls.train = dls.train.new(shuffle=True, bs=4)  # Small batch size for quick iterations
        dls.valid = dls.valid.new(bs=4)
        max_epochs = 1  # Only 1 epoch for quick debugging
    elif not quick_debug:
        max_epochs = 100  # Normal training configuration
        print("Full training mode enabled.")


    # Adjust learning rate
    inf_learner = inf_model.get_learner(dls, criterion, accuracy)

    # Find the best learning rate
    inf_learner.lr_find()  # method will help you visualize how the learning rate affects the loss function.

    # Get the recommended learning rate
    # The suggested learning rate is typically the one found in the plot where the loss starts to decrease significantly.
    best_lr = inf_learner.recorder.lr_min  # This gives you the learning rate corresponding to the minimum loss


    # Apply Early Stopping
    early_stopping = EarlyStopping(patience=patience)

    # Apply Dropout and L2 Regularization to prevent overfitting
    inf_model.model = nn.Sequential(
        inf_model.model,
        nn.Dropout(p=0.5),  # 50% dropout rate
        nn.Linear(inf_model.model[-1].out_features, inf_model.num_classes),
        nn.Softmax(dim=1)
    )
    inf_model.optimizer = torch.optim.Adam(inf_model.parameters(), weight_decay=1e-4)  # L2 regularization

    print(f"Training dataset size: {len(dls.train_ds)}")
    print(f"Validation dataset size: {len(dls.valid_ds)}")
    print(f"Training DataLoader size (number of batches): {len(dls.train)}")
    print(f"Validation DataLoader size (number of batches): {len(dls.valid)}")
    print(f"Batch size during training: {dls.train.bs}")
    print(f"Batch size during validation: {dls.valid.bs}")

    # Print sample data and labels from one batch for sanity check
    xb_train, yb_train = next(iter(dls.train))
    print(f"Sample training batch shape: {xb_train.shape}")
    print(f"Sample training labels shape: {yb_train.shape}")

    xb_valid, yb_valid = next(iter(dls.valid))
    print(f"Sample validation batch shape: {xb_valid.shape}")
    print(f"Sample validation labels shape: {yb_valid.shape}")

    start_time = time.time()
    # Start training for a fixed number of epochs
    for epoch in range(1, max_epochs + 1):
        print(f"Starting epoch {epoch}...")

        # Train the model for one epoch
        inf_learner.fit_one_cycle(1, lr_max=best_lr)  # Train for one epoch

        # Monitor validation loss for early stopping
        val_loss = inf_learner.validate()[0]  # Get the validation loss

        print(f"Epoch {epoch}: Validation Loss: {val_loss:.4f}")

        # Check for early stopping
        if early_stopping(val_loss):
            print(f"Early stopping triggered after epoch {epoch}.")
            break

    end_time = time.time()
    print_time(start_time, end_time, "CNN model training time")

    return inf_model


# Function to save trained model
def save_trained_model(inf_model, dataset_name):
    trained_model_file_name = dataset_name + '_trained_model.pth'
    trained_weights_path = os.path.join(os.getcwd(), 'data', 'output', trained_model_file_name).replace('/',
                                                                                                        get_path_separator())
    print(" ===> Saving trained model to ", trained_weights_path)
    torch.save(inf_model.state_dict(), trained_weights_path)
    print(" ===> Saving terminal output to ", trained_weights_path)
    terminal_out(trained_weights_path)
    return trained_weights_path


# Placeholder for patient data collection logic
def collect_patient_data():
    patient_data = {}  # Example dictionary to store patient data
    print("Collecting patient data...")
    return patient_data


# Placeholder for logic to adjust training parameters
# def adjust_training_parameters():
#    print("Adjusting training parameters...")
#    return {"epochs": 10, "learning_rate": 0.001}


def main():
    train_dataloaders = {}
    print("Starting the main function...")

    # Collect patient data age, sex, systolic blood pressure (SBP), smoking, urinary protein, and HbA1c level as positively associated with the risk
    patient_data = collect_patient_data()  # optional could involve bot or multivariant analysis

    # Download datasets into DATASET_PATH
    kaggle_loader = download_datasets(DATASET_PATH, DATASETS)
    train_dataloaders = load_and_prepare_data(kaggle_loader, dataset_train_structure_resized15_19)

    # Check if CUDA is available and set the device accordingly
    device = check_cuda_availability()

    num_dr_classes = 5  # 0 - No DR, 1 - Mild, 2 - Moderate, 3 - Severe, 4 - Proliferative DR

    pretrained_model = PretrainedEyeDiseaseClassifier(num_classes=num_dr_classes, pretrained_model=pretrained_models[0])
    criterion = nn.CrossEntropyLoss()

    inf_model = EyeDiseaseClassifier(num_classes=num_dr_classes)
    inf_model.to(device)
    pretrained_model.to(device)

    for key, dls in train_dataloaders.items():
        dataset_name = os.path.basename(key).split('.')[0]
        skip_pretrained = True  # Ensure that pretrained model is used

        if not skip_pretrained:
            # Create a timestamp for the filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            pretrained_model_file_name = f"{dataset_name}{timestamp}_pretrained_model.pth"
            pretrained_weights_path = (os.path.join(os.getcwd(), 'data', 'output', pretrained_model_file_name).
                                       replace('/', get_path_separator()))
            print(" \n ===>  Looking for pretrained model here", pretrained_weights_path)
            pretrained_model = train_pretrained_model(pretrained_model, dls, pretrained_weights_path)
        else:
            print("Skipping pretrained model training.")

            # Assuming all necessary libraries are imported
            # Define the configurations as a list of tuples (batch_size, max_epoch)
            configurations = [
                (4, 1),  # Case 1
                (8, 1),  # Case 2
                (16, 1),  # Case 3
                (16, 50),  # Case 4
                (8, 50),  # Case 5
                (4, 50)  # Case 6
            ]

            # Iterate through each configuration
            for bs, max_epoch in configurations:
                print(f"\nRunning training and Validation with Batch Size: {bs} and Max Epoch: {max_epoch}")

                # Update the DataLoader with the current batch size
                dls.train.bs = bs
                dls.valid.bs = bs  # Assuming you want to keep the validation batch size the same

                print("Going to train CNN model...")
                print(f"Training dataset size: {len(dls.train_ds)}")
                print(f"Training DataLoader size: {len(dls.train)}")  # Corrected line
                # Corrected line to access batch size from train DataLoader
                print(f"Batch size during training: {dls.train.bs}")
                # Added for validation batch size
                print(f"Batch size during validation: {dls.valid.bs}")

                #quick_debug = False, True or None not set
                inf_model = train_inference_model(inf_model, dls, criterion, None, patience=5, max_epoch=max_epoch)

                trained_weights_path = save_trained_model(inf_model, dataset_name)
                print(f"Validation dataset size: {len(dls.valid_ds)}")
                print(f"Validation DataLoader size: {len(dls.valid)}")  # Added for validation DataLoader size
                print(f"Batch size during evaluation: {dls.valid.bs}")  # Added for validation batch size

                print(f"\nEvaluating the inference model on dataset: {dataset_name}")
                inf_model.evaluate_model(dls)

                # Print additional information after evaluation
                print("Model evaluation complete for dataset:", dataset_name)


# Call the main function
if __name__ == "__main__":
    main()

# Function Descriptions
# download_datasets(dataset_path, datasets): Downloads datasets from Kaggle into the specified path and returns the Kaggle data loader instance.
#
# load_and_prepare_data(kaggle_loader, dataset_structure): Loads and prepares the training data by constructing full paths for labels and images, returning a dictionary of DataLoaders.
#
# check_cuda_availability(): Checks if CUDA is available and sets the computation device accordingly.
#
# train_pretrained_model(pretrained_model, dls, pretrained_weights_path): Trains the pretrained model if the weights do not already exist, saves the model weights, and evaluates the model.
#
# train_inference_model(inf_model, dls, criterion): Trains the inference model using the provided DataLoader and returns the trained model.
#
# save_trained_model(inf_model, dataset_name): Saves the trained inference model's weights to a specified path and returns the path.
#
# main(): Orchestrates the execution of the training pipeline by calling the above functions to download datasets, prepare data, train models, and save weights.