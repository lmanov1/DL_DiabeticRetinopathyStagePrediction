import os
from code.models.train_model import (
    EyeDiseaseClassifier,
    PretrainedEyeDiseaseClassifier,
    load_model_from_pkl,
    load_model_from_pth,
)
from code.models.train_model import (
    transform_sizes,
    MODEL_FORMAT,
    pretrained_models,
    num_dr_classes,
)

from code.data import data_preparation as DataPrep
from code.data import Dataloader as KaggleDataLoader
from code.data.Dataloader import download_from_kaggle , DATASET_NAMES, DATASET_PATH , ALL_DATASETS
from code.Util.MultiPlatform import *
import torch
import torch.nn as nn
import torch.optim as optim
from fastai.vision.all import *
from torch.utils.data import WeightedRandomSampler
import time
from code.config import *
import argparse

# PretrainedEyeDiseaseClassifier defines pretrained vision model , may be one of the defined in train_model.py
# DataBlock and DataLoaders handle the data processing and augmentation.
# The Learner class from fastai handles training and evaluation seamlessly.

def get_dataloaders(task_name, dataset_structure, dataset_dir, imaging_format):
    """Returns the DataLoaders for the relevant dataset task (train, valid, or test)"""
    dataloaders = {}
    for dataset in dataset_structure:
        print(dataset)
        print("---------------------------------------")
        dataset["labels"] = os.path.join(dataset_dir.strip(""), dataset["labels"])
        dataset["images"] = str(os.path.join(dataset_dir, dataset["images"]))
        print(f"{dataset['labels']} \n {dataset['images']}")
        print("=====================================")

        # determine the transform size based on the model
        transform_size = transform_sizes[current_model_name]
        dataloader = DataPrep.DataPreparation(
            dataset["labels"], dataset["images"], imaging_format, transform_size , split_data_dataloaders
        )
        dataloader.load_data()
        dls = dataloader.get_dataloaders()
        dls.show_batch()
        dataloaders[dataset["labels"]] = dls
        print("=====================================")

    print(f"{task_name}_dataloaders: {dataloaders}")
    return dataloaders

# Assigning Weights Based on Class Imbalance (deriving class counts for each class from a dataloader)
def calculate_class_weigths(xdataloaders: DataLoaders):
    """
    Calculate the weights for learner's loss function and update the DataLoader with the class weights.
    """
    label_counts = {}
    # Assuming labels are in the second column
    labels = xdataloaders.train_ds.items.iloc[:, 1]
    counts = labels.value_counts()
    label_counts = dict(zip(counts.index, counts.values))
    print(f"===========> Label counts for {xdataloaders} : {label_counts}")
    total = sum(label_counts.values())
    weights = {k: total / v for k, v in label_counts.items()}
    # Convert weights to a list in the order of your class indices
    weight_list = [weights[k] for k in sorted(weights.keys())]
    class_weights = (
        torch.FloatTensor(weight_list).cuda()
        if torch.cuda.is_available()
        else torch.FloatTensor(weight_list)
    )
    print(f"===========> Class weigth: {weight_list}")
    print(f"===========> Class weigth tensor: {class_weights}")
    # Update data loaders with class weights
    # Use the WeightedRandomSampler for the train DataLoader
    train_dl = xdataloaders.train.new(dl_kwargs={'sampler': WeightedRandomSampler(weights=class_weights, num_samples=len(class_weights),replacement=True)})
    xdataloaders.train = train_dl

    return class_weights


def evaluate_model_on(learner, dls, model , task_name=None):
    """
    Evaluate the model on the specified DataLoader.
    """
    print("\n---------------------------------------------------------------------------")
    print(f"==> Evaluating model on {task_name} dataset")
    results = None
    if task_name == "val":
        if split_data_dataloaders == True:
            results = learner.validate(dl=dls.valid)
        else:
            results = learner.validate(dl=dls.train)
        model.evaluate_model_on(learner, dls, "val")

    elif task_name == "test":
        results = learner.validate(dl=dls.train)
        model.evaluate_model_on(learner, dls, "test")
    else:
        raise ValueError("Invalid task name. Use 'val' or 'test'.")

    loss, accuracy = results[0], results[1]
    print(f"{task_name.capitalize()} Loss: {loss:.4f}")
    print(f"{task_name.capitalize()} Accuracy: {accuracy:.4f}")


# ==============================================================================

def main():
    """
    Main function to train or evaluate a model for diabetic retinopathy stage prediction.
    This function handles the following tasks:
    1. Parses command-line arguments to determine if a pre-trained model should be loaded.
    2. If a model file is provided, loads the model from the specified file (.pkl or .pth format).
    3. If no model file is provided, downloads the dataset from Kaggle and prepares data loaders for training, validation, and testing.
    4. Handles data imbalance by calculating class weights if specified.
    5. Checks if the model needs to be trained or if a pre-trained model already exists.
    6. Trains the model if necessary and saves the trained model in both .pth and .pkl formats.
    7. Evaluates the model on validation and test datasets if available.
    Command-line Arguments:
    --model_file (str): Path to the model file to load.
    Raises:
    ValueError: If the provided model file format is not supported.
    """
    train_dataloaders = {}
    current_dataset = None
    Epochs = 2
    valid_dataloaders = None
    test_dataloaders = None
    # for key, dls in train_dataloaders.items():
    val_key, val_dls, test_key, test_dls, train_key, train_dls = (
        None,
        None,
        None,
        None,
        None,
        None,
    )
    testset_evaluation = testset_evaluation_if_exists
    
    print(f"Using device: {device}")
    parser = argparse.ArgumentParser(description="Train or evaluate a model for diabetic retinopathy stage prediction.")
    parser.add_argument('--model_file', type=str, help='Path to the model file to load')
    parser.add_argument('--dataset', type=str, choices=DATASET_NAMES, default=DATASET_NAMES[1], help='Name of the dataset to use')
    parser.add_argument('--usetest', type=bool, choices=[False, True], default=False, help='Evaluate the model on the test dataset (the one specified with --dataset ) if available')
    parser.add_argument('--torch_save_mode', type=str, choices=['weights', 'huggingface_weights' , 'full'], default='weights', help='Specify pretraining model loading mode derived from how it was saved. Relevant for torch only, default is weights only')    
    args = parser.parse_args()

    if args.dataset:
        current_dataset = args.dataset    
    if args.usetest:
        testset_evaluation = args.usetest
    if args.model_file:
        if args.model_file.endswith('.pkl'):
            pretrained_model = load_model_from_pkl(args.model_file)
        elif args.model_file.endswith('.pth'):
            torch_save_mode = 'weights'
            if args.torch_save_mode:
                torch_save_mode = args.torch_save_mode
            pretrained_model = load_model_from_pth(args.model_file, torch_save_mode)
            
        else:
            raise ValueError("Unsupported model file format. Use .pkl or .pth")
        pretrained_learn = pretrained_model.get_learner()
        pretrained_learn.model.eval()
        print(f"Loaded model from {args.model_file}")
    else:           
        current_dataset = DATASET_NAMES[1]
    
    # Download the dataset from Kaggle if needed
    dataset_dir = download_from_kaggle(current_dataset, DATASET_PATH)
    print(f"Dataset directory: {dataset_dir}")                       

    # Get the DataLoaders for the training and validation;  Test dataset will be created later if needed
    train_dataloaders = get_dataloaders(
        "train",
        ALL_DATASETS[current_dataset]["train"],
        dataset_dir,
        ALL_DATASETS[current_dataset]["imaging_format"],
    )
    train_key, train_dls = list(train_dataloaders.items())[0]

    if split_data_dataloaders == False:
        # Split the training DataLoader datasets into train and validation DataLoaders
        # instead of creating separate DataLoaders for each dataset
        if ALL_DATASETS[current_dataset]["val"] is not None:
            valid_dataloaders = get_dataloaders(
                "val",
                ALL_DATASETS[current_dataset]["val"],
                dataset_dir,
                ALL_DATASETS[current_dataset]["imaging_format"],
            )
            val_key, val_dls = list(valid_dataloaders.items())[0]        

        # Create model class and load pretrained weights or train the model
        # Calculate weigths for loss function - this should address possible data imbalance issues with the dataset
        # Yes our data , as turns out - is higthly imbalanced , with majority of cases labeled with severity 0 (No DR)
        if handle_data_disbalance == True:
            class_weights = calculate_class_weigths(train_dls)
        else:
            class_weights = None


        # Check if model file exists - train now if not found or load saved model if exists
        need_to_train = True
        need_to_train, pretrained_torch_path, pretrained_pkl_path , model_type = (
            check_need_to_train_model(train_key, ALL_DATASETS[current_dataset]["name"], current_model_name)
        )        

        if need_to_train:
            pretrained_model = PretrainedEyeDiseaseClassifier(
            num_classes=num_dr_classes,
            model_name=current_model_name
            ,label_weights=class_weights
            )
            pretrained_model.to(device)
            start_time = time.time()
            pretrained_model.train_model(train_dls, epochs=Epochs)
            end_time = time.time()
            print_time(start_time, end_time, "Elapsed fine-tuning time")
            pretrained_learn = pretrained_model.get_learner()
            # ==============================================================================
            # Save the model (torch and pickle)            
            best_model_state = deepcopy(
                pretrained_model.get_learner().model.state_dict()
            )  # don't save the pointer in memory but the entire object
            torch.save(best_model_state, pretrained_torch_path)
            

            print(" ===>  Saving pretrained model to ", pretrained_torch_path)
            # Save the model weigths (in pkl format - unsecure but backward compatible in many cases)
            print(" ===>  Exporting pretrained model to ", pretrained_pkl_path)
            pretrained_learn.export(pretrained_pkl_path)
        
        else:  # Load the pretrained model            
            if model_type == '.pkl':
                print(f" ===>  Pretrained model already exists at {pretrained_pkl_path}") 
                pretrained_model = load_model_from_pkl(pretrained_pkl_path)
                                     
            elif model_type == '.pth':
                print(f" ===>  Pretrained model already exists at {pretrained_torch_path}") 
                pretrained_model = load_model_from_pth(pretrained_torch_path)
                           
            pretrained_learn = pretrained_model.get_learner()        
            pretrained_learn.model.eval() # Set dropout and batch normalization layers to evaluation mode before running inference

    # ==============================================================================
    # Evaluate the model on the validation and test datasets
    if split_data_dataloaders == True:
        evaluate_model_on(pretrained_learn, train_dls, pretrained_model , task_name='val')

    elif val_dls is not None:
        evaluate_model_on(pretrained_learn, val_dls, pretrained_model , task_name='val')
    
    # final evaluation on the test dataset if exists and configured to do so
    if testset_evaluation == True and ALL_DATASETS[current_dataset]["test"] is not None:
        test_dataloaders = get_dataloaders(
            "test",
            ALL_DATASETS[current_dataset]["test"],
            dataset_dir,
            ALL_DATASETS[current_dataset]["imaging_format"],
        )
        test_key, test_dls = list(test_dataloaders.items())[0]

    # Final evaluation on the test dataset (always separate from the training dataset)
    if test_dls is not None:
        evaluate_model_on(pretrained_learn, test_dls, pretrained_model , task_name='test')

# Call the main function
if __name__ == "__main__":
    main()

