import os
#from models import inference_model as Neural_Net
from models.train_model import EyeDiseaseClassifier , PretrainedEyeDiseaseClassifier , pretrained_models
from data import data_preparation as DataPrep
#from data import data_preprocessing as data_preprocessing
from data import Dataloader as KaggleDataLoader
import torch 
import torch.nn as nn
import torch.optim as optim
from fastai.vision.all import *
import time

# Define the global variable for the dataset path
DATASET_PATH = '/path/to/your/dataset'

# Choose here the dataset(s) you want to download (https://www.kaggle.com/search?q=APTOS+2019+Blindness+Detection+Dataset+in%3Adatasets)
# Dataset names can be found under the  https://www.kaggle.com/datasets page
DATASET_NAME_resized15_19 = 'benjaminwarner/resized-2015-2019-blindness-detection-images'   # 18.75 GB  
DATASET_NAME_aptos19 = 'mariaherrerot/aptos2019'   # 8.6GB
DATASET_PATH = 'data/raw/'

# Define the dataset structure .. to be continued for validation and test datasets
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

# EyeDiseaseClassifier defines CNN model.
# PretrainedEyeDiseaseClassifier defines pretrained vision model , either resnet18 or vgg16 model.
# DataBlock and DataLoaders handle the data processing and augmentation.
# The Learner class from fastai handles training and evaluation seamlessly.

def main():
    train_dataloaders = {}
    print("Starting the main function...")
    # Download datasets into DATASET_PATH
    print("Downloading datasets...")
    kaggle_loader = KaggleDataLoader.KaggleDataDownLoader(DATASET_PATH , DATASETS[0])
    print(f"Dataset downloaded into {kaggle_loader.dataset_dir}")
    print("Loading train aptos 19 dataset...")

    # Construct the full paths for labels and images
    for dataset in dataset_train_structure_resized15_19:
        print("---------------------------------------")
        print(dataset)
        dataset['labels'] = os.path.join(kaggle_loader.dataset_dir.strip(''), dataset['labels'])
        dataset['images'] = str(os.path.join(kaggle_loader.dataset_dir, dataset['images']))
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

    pretrained_model = PretrainedEyeDiseaseClassifier(num_classes=num_dr_classes, pretrained_model= pretrained_models[0])    
    criterion = nn.CrossEntropyLoss() #CrossEntropyLossFlat
    #pretr_optimizer = optim.Adam(pretrained_model.parameters(), lr=0.001)

    inf_model = EyeDiseaseClassifier(num_classes=num_dr_classes) 
    #inf_optimizer = optim.Adam(inf_model.parameters(), lr=0.001)
    
    inf_model.to(device)
    pretrained_model.to(device)

    for key, dls in train_dataloaders.items():

        #1. Train the (pretrained) model
        start_time = time.time()
        pretrained_model.train_model(dls, epochs=10)
        end_time = time.time()
        print(f"==> Pretrained model training time: {end_time - start_time} seconds")
        # Save the pretrained model weights
        torch.save(pretrained_model.state_dict(), str(str(key)+ '_pretrained_model.pth'))
        # Evaluate the model
        pretrained_model.evaluate_model(dls)  

        # 2. Train the (inference) model
        # Load the pretrained weights into the main model
        #inf_model.load_state_dict(torch.load(str(str(key)+ '_pretrained_model.pth')))
        learn = Learner(dls, inf_model, loss_func=criterion, metrics=accuracy)
        #Training (fit_one_cycle): We use fit_one_cycle for training the model from scratch, as it’s more suited 
        # for models without pretrained weights.
        start_time = time.time()
        learn.fit_one_cycle(10)
        end_time = time.time()
        print(f"==> CNN model training time: {end_time - start_time} seconds")
        torch.save(pretrained_model.state_dict(), str(str(key)+ '_inference_model.pth'))
        # Evaluate the model  - Uses ClassificationInterpretation to plot the confusion matrix and the top losses.
        interp = ClassificationInterpretation.from_learner(learn)
        interp.plot_confusion_matrix()
        interp.plot_top_losses(k=9, nrows=3)                              

    #print("Model and data are set up and ready!")


# Call the main function
if __name__ == "__main__":
    main()
