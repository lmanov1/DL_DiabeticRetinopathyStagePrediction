import os
#from models import inference_model as Neural_Net
from code.models.train_model import EyeDiseaseClassifier , PretrainedEyeDiseaseClassifier , load_model_from_pkl  ,pretrained_models , num_dr_classes ,MODEL_FORMAT
from code.data import data_preparation as DataPrep
#from data import data_preprocessing as data_preprocessing
from code.data import Dataloader as KaggleDataLoader
import torch 
import torch.nn as nn
import torch.optim as optim
from fastai.vision.all import *
from Util.MultiPlatform import *
import time

# Define the dataset names and paths
# Choose here the dataset(s) you want to download (https://www.kaggle.com/search?q=APTOS+2019+Blindness+Detection+Dataset+in%3Adatasets)
# Dataset names can be found under the  https://www.kaggle.com/datasets page
DATASET_NAME_resized15_19 = 'benjaminwarner/resized-2015-2019-blindness-detection-images'   # 18.75 GB  
DATASET_NAME_aptos19 = 'mariaherrerot/aptos2019'   # 8.6GB
DATASET_PATH = 'data/raw/'

# Define the datasets structure .. here are tests and train datasets
dataset_train_structure_resized15_19 = [
    # {
    #     'labels': 'labels/trainLabels15.csv',
    #     'images': 'resized train 15'        # 6.66GB on disk , 35126 files   
    # },
    {
        'labels': 'labels/trainLabels19.csv', 
        'images': 'resized train 19'  # 630MB on disk , 3662 files      
    }
]

DATASETS = [ DATASET_NAME_resized15_19 , DATASET_NAME_aptos19] 

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



def get_saved_model_name(dataset_name, model_type, model_name_prefix = ""):
    """Returns the name of the saved model file."""

    if model_type not in MODEL_FORMAT:
        raise ValueError(f"Unsupported model format. Choose one of: {MODEL_FORMAT}")    
    
    model_file_name = model_name_prefix + dataset_name + '_export.' + model_type    
    return os.path.join(os.getcwd(), 'data', 'output', model_file_name).replace('/', get_path_separator())
    

#==============================================================================

def main():
    train_dataloaders = {}
    print("Starting the main function...")
    
    # Load the datasets into the DataLoaders (from local storage)    
    for dataset in dataset_train_structure_resized15_19:
        
        print(dataset)
        print("---------------------------------------")
        dataset_dir = download_from_kaggle(DATASETS[0], DATASET_PATH)
        dataset['labels'] = os.path.join(dataset_dir.strip(''), dataset['labels'])
        dataset['images'] = str(os.path.join(dataset_dir, dataset['images']))
        #csv_path, img_folder, label_col='label', image_col = 'image_path',, valid_pct=0.2, batch_size=32, seed=42
        print(f"{dataset['labels']} \n {dataset['images']}")
        print("=====================================")
        dataloader = DataPrep.DataPreparation(dataset['labels'],dataset['images'])        
        dataloader.load_data()
        # dataloader.normalize_data()
        # dataloader.augment_data()
        dls = dataloader.get_dataloaders()           
        dls.show_batch()
        train_dataloaders[dataset['labels']] = dls
        print("=====================================")

    print(f"train_dataloaders: {train_dataloaders}")
    
    # Check if CUDA is available and set the device accordingly
    print("Checking for CUDA availability...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
   
    pretrained_model = PretrainedEyeDiseaseClassifier(num_classes=num_dr_classes, pretrained_model= pretrained_models[0])    
    criterion = nn.CrossEntropyLoss() #CrossEntropyLossFlat
    #pretr_optimizer = optim.Adam(pretrained_model.parameters(), lr=0.001)

    inf_model = EyeDiseaseClassifier(num_classes=num_dr_classes) 
    #inf_optimizer = optim.Adam(inf_model.parameters(), lr=0.001)
    
    inf_model.to(device)
    pretrained_model.to(device)
    
    for key, dls in train_dataloaders.items():
        #1. Fine tune the (pretrained) model
        print("Current directory:", os.getcwd())
        # Extract the key from the path
        dataset_name = os.path.basename(key).split('.')[0]
        print("Dataset name:", dataset_name)        
        pretrained_pkl  = get_saved_model_name(dataset_name, 'pkl', 'pretrained_')
        pretrained_weigths_path  = get_saved_model_name(dataset_name, 'pth', "pretrained_")
        print(" \n ===>  Looking for pretrained model here ", pretrained_weigths_path , pretrained_pkl)      # 513 MB
        
        # very heavy run - about 8 hours on 100% GPU - lets not run it again
        if not os.path.exists(pretrained_weigths_path):
            print("Pretrained model not found - training now...")
            # win fix - Ensure directory exists
            directory = os.path.dirname(pretrained_weigths_path)
            if not os.path.exists(directory):
                os.makedirs(directory)


            start_time = time.time()
            pretrained_model.train_model(dls, epochs=10)
            end_time = time.time()
            print_time(start_time , end_time , "Pretrained model training time")
            # Evaluate the model
            pretrained_model.evaluate_model(dls)  
            # Save the pretrained model weights             
            best_model_state = deepcopy(pretrained_model.state_dict())  # don't save the pointer in memory but the entire object
            torch.save(best_model_state, pretrained_weigths_path)
            print(" ===>  Saving pretrained model to ", pretrained_weigths_path)                                     
            
            # export the model
            pretrained_learn = pretrained_model.get_learner(dls, criterion, accuracy)            
            print(" ===>  Exporting pretrained model to ", pretrained_pkl)           
            pretrained_learn.export(pretrained_pkl)

        else:           
            print(" ===>  Pretrained model already exists at ", pretrained_weigths_path)
            # inf_learner = load_model_from_pkl(pretrained_pkl)
            # inf_learner.model.eval()            
            # pred, idx, probs = inf_learner.predict("data/raw/benjaminwarner/resized-2015-2019-blindness-detection-images/resized train 19/664b1f9a2087.jpg")            
            # print(f"Prediction: {pred}; Probability: {probs[idx]:.04f}")
            # print(inf_learner.dls.vocab)
        
        print("Going to train CNN model...")                
        # 2. Train model      
        inf_learner = inf_model.get_learner(dls, criterion, accuracy)
        #Training (fit_one_cycle): We use fit_one_cycle for training the model from scratch, as it’s more suited 
        # for models without pretrained weights.
        start_time = time.time()
        inf_learner.fit_one_cycle(10)
        end_time = time.time()
        print_time(start_time , end_time , "CNN model training time")
        inf_model.evaluate_model(dls)    
        # Save the trained model weights        
        trained_weigths_path  = get_saved_model_name(dataset_name, 'pth', "trained_")        
        print(" ===> Saving CNN train model to ", trained_weigths_path)
        best_model_state = deepcopy(inf_model.state_dict())  # don't save the pointer in memory but the entire object
        torch.save(best_model_state, trained_weigths_path )
        print(" ===>  Saving pretrained model to ", trained_weigths_path) 

        ptrained_learn = inf_model.get_learner(dls, criterion, accuracy)   
        pretrained_pkl  = get_saved_model_name(dataset_name, 'pkl', 'trained_')
        print(" ===>  Exporting pretrained model to ", pretrained_pkl)           
        ptrained_learn.export(pretrained_pkl)
          

# Call the main function
if __name__ == "__main__":
    main()
