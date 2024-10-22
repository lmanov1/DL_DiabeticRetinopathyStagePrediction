import os
#from code.models import inference_model as Neural_Net
from code.models.train_model import EyeDiseaseClassifier , PretrainedEyeDiseaseClassifier , load_model_from_pkl  , load_model_from_pth , pretrained_models , num_dr_classes ,MODEL_FORMAT
from code.data import data_preparation as DataPrep
#from code.data import data_preprocessing as data_preprocessing
from code.data import Dataloader as KaggleDataLoader
from code.Util.MultiPlatform import *
import torch 
import torch.nn as nn
import torch.optim as optim
from fastai.vision.all import *
import time

# Define the dataset names and paths
# Choose here the dataset(s) you want to download (https://www.kaggle.com/search?q=APTOS+2019+Blindness+Detection+Dataset+in%3Adatasets)
# Dataset names can be found under the  https://www.kaggle.com/datasets page
DATASET_NAME_resized15_19 = 'benjaminwarner/resized-2015-2019-blindness-detection-images'   # 18.75 GB  
DATASET_NAME_aptos19 = 'mariaherrerot/aptos2019'   # 8.6GB
DATASET_PATH = 'data/raw/'

# Define the datasets structure .. here are tests and train datasets
dataset_train_structure_resized15_19 = [
    {
        'labels': 'labels/trainLabels15.csv',
        'images': 'resized train 15'        # 6.66GB on disk , 35126 files   
    }
    # ,
    # {
    #     'labels': 'labels/trainLabels19.csv', 
    #     'images': 'resized train 19'  # 630MB on disk , 3662 files      
    # }
]

dataset_test_structure_resized15_19 = [
    {
        'labels': 'labels/testLabels15.csv',    # 53577 
        'images': 'resized test 15'        # 11GB on disk , 53577 files   
    }
    #,
    # {
    #     'labels': 'labels/testLabels19.csv', 
    #     'images': 'resized train 19'  # 630MB on disk , 3662 files      
    # }
]

dataset_train_structure_aptos19 = [
    {
        'labels': 'train_1.csv',        # 2930 rows
        'images': 'train_images/train_images'        # 6.4 GB 2930 images 
    }
]
dataset_test_structure_aptos19 = [  
    {
        'labels': 'test.csv',           #  need to crop as file has 366 rows with data and all the rest is empty
        'images': 'test_images/test_images'         # 796 MB 366 images
    }
]
dataset_val_structure_aptos19   = [ 
    {
        'labels': 'valid.csv',     #  366 rows
        'images': 'val_images/val_images'     # 902 MB   366 images   
    }
]

DATASET_NAMES = [ DATASET_NAME_resized15_19 , DATASET_NAME_aptos19] 

# Create a data structure to map each dataset to its associated train and test structures (and val if exists)
datasets = {
    DATASET_NAME_resized15_19: {
        'train': dataset_train_structure_resized15_19,
        'test': dataset_test_structure_resized15_19,
        'val': None,
        'imaging_format': '.jpg',
        'name': 'resized15_19'
    },
    # Add other datasets and their structures here if needed
    DATASET_NAME_aptos19: {
        'train': dataset_train_structure_aptos19,
        'test': dataset_test_structure_aptos19,
        'val': dataset_val_structure_aptos19,
        'imaging_format': '.png',
        'name': 'aptos19'
    }
}



# EyeDiseaseClassifier defines CNN model.
# PretrainedEyeDiseaseClassifier defines pretrained vision model , either resnet18 or vgg16 model.
# DataBlock and DataLoaders handle the data processing and augmentation.
# The Learner class from fastai handles training and evaluation seamlessly.


def download_from_kaggle(dataset_name, dataset_path , kaggle_json_path=None):
    """
    Downloads a dataset from Kaggle and saves it to the specified path.
    Args:
        dataset_name (str): The name of the dataset to download from Kaggle.
        dataset_path (str): The local path where the dataset should be saved.
        kaggle_json_path (str, optional): The path to the Kaggle JSON file for authentication. Defaults to None.
    Returns:
        str: The directory where the dataset has been downloaded.
    """
    print(f"Downloading dataset {dataset_name} into {dataset_path}")
    kaggle_downloader = KaggleDataLoader.KaggleDataDownLoader(dataset_path , dataset_name, kaggle_json_path)    
    return kaggle_downloader.dataset_dir

def get_dataloaders(task_name, dataset_structure, dataset_dir, imaging_format):
    """Returns the DataLoaders for the relevant dataset task (train, valid, or test)"""
    dataloaders = {}
    for dataset in dataset_structure:
        print(dataset)
        print("---------------------------------------")
        dataset['labels'] = os.path.join(dataset_dir.strip(''), dataset['labels'])
        dataset['images'] = str(os.path.join(dataset_dir, dataset['images']))
        print(f"{dataset['labels']} \n {dataset['images']}")
        print("=====================================")
        dataloader = DataPrep.DataPreparation(dataset['labels'], dataset['images'], imaging_format)
        dataloader.load_data()      
        dls = dataloader.get_dataloaders()
        dls.show_batch()
        dataloaders[dataset['labels']] = dls
        print("=====================================")

    print(f"{task_name}_dataloaders: {dataloaders}")
    return dataloaders

def check_need_to_train_model(dls_key , dataset_name, model_name):
    """Checks if a model needs to be trained or if a saved model already exists."""
        #1. Fine tune the (pretrained) model
    print("Current directory:", os.getcwd())
    # dataset name + specific data set name from currently processed data loaders       
    # i.e. resnet50_aptos19train_1.pkl/pth
    print("Dataset name:", dataset_name)
    dataset_name = str(dataset_name + os.path.basename(dls_key).split('.')[0])
    
    pretrained_pkl_path  = get_saved_model_name(dataset_name, 'pkl', str(model_name + "_"))
    pretrained_torch_path  = get_saved_model_name(dataset_name, 'pth', str(model_name + "_"))
    print(" \n ===>  Looking for pretrained model here ", pretrained_torch_path , pretrained_pkl_path)      # 513 MB
    
    # Depends on dataset , but in general it will be a very heavy run - 
    # about 8 hours on 100% GPU - lets not run it again if we have the model      
    if not os.path.exists(pretrained_torch_path):
        print("Trained model not found - fine-tuning now...")
        # win fix - Ensure directory exists
        directory = os.path.dirname(pretrained_torch_path)
        if not os.path.exists(directory):
            os.makedirs(directory)   
        return True , pretrained_torch_path,  pretrained_pkl_path
    else:           
        print(" ===>  Trained model already exists at ", pretrained_torch_path)
        return False , pretrained_torch_path,  pretrained_pkl_path

    

def get_saved_model_name(dataset_name,  output_format, model_name_prefix = ""):
    """Returns the name of the saved model file."""

    if output_format not in MODEL_FORMAT:
        raise ValueError(f"Unsupported model format. Choose one of: {MODEL_FORMAT}")    
    
    model_file_name = model_name_prefix + dataset_name +  '.' + output_format    
    return os.path.join(os.getcwd(), 'data', 'output', model_file_name).replace('/', get_path_separator())
    

#==============================================================================

def main():
    train_dataloaders = {}
    print("Starting the main function...")
    Current_model_name = 'resnet50'
    Epochs = 10
    Current_dataset = DATASET_NAMES[1]
    dataset_dir = download_from_kaggle(DATASET_NAMES[1], DATASET_PATH)
    print(f"Dataset directory: {dataset_dir}")

     # Check if CUDA is available and set the device accordingly
    print("Checking for CUDA availability...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    #three DataLoaders are created: one for the training dataset and one for each of the validation and test datasets.
    # Note that for validation and test DataLoaders, we directly use .train_dl since no splitting is applied.
    train_dataloaders = get_dataloaders('train', datasets[Current_dataset]['train'], dataset_dir, datasets[Current_dataset]['imaging_format'])   
    valid_dataloaders = get_dataloaders('val', datasets[Current_dataset]['val'], dataset_dir, datasets[Current_dataset]['imaging_format'])       
    test_dataloaders = get_dataloaders('test', datasets[Current_dataset]['test'], dataset_dir, datasets[Current_dataset]['imaging_format'])
    
    pretrained_model = PretrainedEyeDiseaseClassifier(num_classes=num_dr_classes, model_name = Current_model_name)    
    pretrained_model.to(device)    

    #for key, dls in train_dataloaders.items():
    val_key , val_dls = list(valid_dataloaders.items())[0]    
    test_key , test_dls = list(test_dataloaders.items())[0]
    key , train_dls = list(train_dataloaders.items())[0]
    need_to_train = True
    need_to_train , pretrained_torch_path,  pretrained_pkl_path = \
        check_need_to_train_model(key , datasets[Current_dataset]['name'], Current_model_name)
    
    if need_to_train:
            start_time = time.time()
            pretrained_model.train_model(train_dls, epochs=Epochs)
            end_time = time.time()
            print_time(start_time , end_time , "Pretrained model fine-tuning time")
                        
            # Save the model weigths (torch format)             
            best_model_state = deepcopy(pretrained_model.state_dict())  # don't save the pointer in memory but the entire object
            torch.save(best_model_state, pretrained_torch_path)
            print(" ===>  Saving pretrained model to ", pretrained_torch_path)                                                 
            # Save the model weigths (in pkl format - unsecure but backward compatible in many cases)
            pretrained_learn = pretrained_model.get_learner()            
            print(" ===>  Exporting pretrained model to ", pretrained_pkl_path)           
            pretrained_learn.export(pretrained_pkl_path)

    else:           
        print(" ===>  Pretrained model already exists at ", pretrained_torch_path)
        # fix learner - should be set to a model which wasn't actually trained since creation
        #pretrained_model = load_model_from_pth(pretrained_model, pretrained_torch_path)
        pretrained_learn = load_model_from_pkl(pretrained_pkl_path)        
        pretrained_model.set_learner(pretrained_learn)     
        pretrained_learn.model.eval()        
        
        pred, idx, probs = pretrained_learn.predict("data/raw/benjaminwarner/resized-2015-2019-blindness-detection-images/resized train 19/664b1f9a2087.jpg")            
        print(f"Prediction: {pred}; Probability: {probs[idx]:.04f}")                    
    
    #==============================================================================
    #The Learner object is updated to include the validation DataLoader 
    pretrained_learn.dls = val_dls
    # Validation: Get validation loss and accuracy
    val_results = pretrained_learn.validate()             
        
    val_loss, val_accuracy = val_results[0], val_results[1]
    print(f'Validation Loss: {val_loss:.4f}')
    print(f'Validation Accuracy: {val_accuracy:.4f}')
    pretrained_model.evaluate_model(pretrained_learn, val_dls)
    #==============================================================================
    # Evaluation on the test dataset
    pretrained_learn.dls = test_dls
    test_results = pretrained_learn.validate()        
    test_loss, test_accuracy = test_results[0], test_results[1]
    
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')    
    pretrained_model.evaluate_model(pretrained_learn, test_dls)
    
    # You can also visualize predictions on the validation set
    pretrained_learn.show_results()
    plt.savefig('results.png')
        
   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # # 2. Train model 
        #  #pretr_optimizer = optim.Adam(pretrained_model.parameters(), lr=0.001)
        #inf_model = EyeDiseaseClassifier(num_classes=num_dr_classes) 
        #inf_optimizer = optim.Adam(inf_model.parameters(), lr=0.001)
        #inf_model.to(device)     
        # inf_learner = inf_model.get_learner(dls, criterion, accuracy)
        # #Training (fit_one_cycle): We use fit_one_cycle for training the model from scratch, as it’s more suited 
        # # for models without pretrained weights.
        # start_time = time.time()
        # inf_learner.fit_one_cycle(10)
        # end_time = time.time()
        # print_time(start_time , end_time , "CNN model training time")
        # inf_model.evaluate_model(dls)    
        # # Save the trained model weights        
        # trained_weigths_path  = get_saved_model_name(dataset_name, 'pth', "trained_")        
        # print(" ===> Saving CNN train model to ", trained_weigths_path)
        # best_model_state = deepcopy(inf_model.state_dict())  # don't save the pointer in memory but the entire object
        # torch.save(best_model_state, trained_weigths_path )
        # print(" ===>  Saving pretrained model to ", trained_weigths_path) 

        # ptrained_learn = inf_model.get_learner(dls, criterion, accuracy)   
        # pretrained_pkl_path  = get_saved_model_name(dataset_name, 'pkl', 'trained_')
        # print(" ===>  Exporting pretrained model to ", pretrained_pkl_path)           
        # ptrained_learn.export(pretrained_pkl_path)

    # End training set , now test set

# Call the main function
if __name__ == "__main__":
    main()

