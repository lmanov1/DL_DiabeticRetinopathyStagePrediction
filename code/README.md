[//]: # (__init__.py file is used to mark a directory as a Python package, making it easier to organize and structure your code.)
  
#### Data Acquisition    
Dataloader.py
##### Class: KaggleDataDownLoader

Methods:      
- __init__(self, dataset_path, dataset_name, kaggle_json_path=None): Initializes the downloader with dataset path, name, and optional Kaggle JSON path.        
- create_kaggle_json(self): Creates the Kaggle API JSON file if it doesn't exist.    
- setup_kaggle_api(self): Sets up the Kaggle API using the JSON file.        
- get_dataset_dir(self): Returns the dataset directory.      

#### Data Preparation     
data_preparation.py        
##### Class: DataPreparation     
This class code is based on fastai API, and optimized  for use with labeled imaging data - utilizes  data block, data loader
######Methods    
- __init__(self, csv_path, img_folder, valid_pct=0.2, batch_size=32, seed=42): - Initializes the data preparation with paths, validation percentage, batch size, and seed.
- load_data(self): Loads data from a CSV file and creates a DataBlock for image processing.
- normalize_data(self): Normalizes the data using the statistics of the training set.
- augment_data(self): Applies data augmentation techniques.
- get_dataloaders(self): Gets the DataLoaders for training and validation.
- show_batch(self, n=9): Shows a batch of images with labels.


#### Model definition     
train_model.py

##### Class: CustomModelMethods
Is a base class that defines methods for training and evaluating a model, using a learner object
Methods:
- train_model - for fine-tuning models with pretrained weigth , using an internal learner object
- evaluate_model - using internal learner object , loss function of CrossEntropyLossFlat and accuracy metrics.
- get_learner(evaluation metrcis) - get an internal learner object
##### Class: PretrainedEyeDiseaseClassifier

This class implements a vision classifier based on a publicly available pre-trained model and is seamlessly integrated with fastai - data block, data loader, and learner. 
This class can be used as a reference point for the performance of a model under development or/and for transfer learning (which utilizes weights and biases of this model on a model under development).
Currently works with resnet18 or vgg16 , but It in general can be any model out of the collection supported by torchvision.models.
Inherits parent classes torch.nn.Module and CustomModelMethods.
Methods:
- __init__(self, num_classes=5, pretrained_model='vgg16'): Initializes the model with the number of classes and the choice of pre-trained model (vgg16 or resnet18).
- forward(self, x): Defines the forward pass of the model.
- set_num_classes(self, num_classes): Sets the number of classes for the classifier layer.

##### Class: EyeDiseaseClassifier    

This is a generic CNN classification model that can be used for different eye disease diagnostics ( based on retina fundus images).This flexibility is due to the configurable number of classes for model use (num_classes = 5 (0..4) in case of Diabetic Retinopathy classification).  
The model can be trained on different datasets, each with its relevant disease-related labels (based on ‘num classes’ parameter which defines the last decision layer shape).     
The class uses Fastai data loaders that allow dataset iteration on (label, image) batches.
Inherits parent classes torch.nn.Module and CustomModelMethods.
Methods
- __init__(self, num_classes=5): Initializes the CNN model with the number of classes.
- forward(self, x): Defines the forward pass of the model.
- set_num_classes(self, num_classes): Sets the number of classes for the classifier layer.

#### Main flow      
Main Function: main()            
- Downloads the datasets. Currently works with        
    `benjaminwarner/resized-2015-2019-blindness-detection-images`         
    See `Define the dataset names and paths` section in code/main.py for more details , there are more available datasets available in Kaggle that we can use.
- Prepares the DataLoaders      
- Trains and evaluates both pre-trained and inference models, saves models under data/output as *.pth (torch model format). These files kept local as it can be too big to be uploaded to github.         
- On subsequent run , once pretrained model (previously fine tuned on specific dataset) is found  under the `data/output/datasetname_pretrained.pth`, it will not be run again.    
 Instead, training a CNN model (which currently trained with learner's fit_one_cycle()) will be run over and over again.     
    
- Uses Fastai's Learner class to handle training and evaluation      
- This flow is the very first draft , to be improved     

#### Util      
MultiPlatform.py            
- get_path_separator(): Determines and returns the appropriate path separator based on the operating system.        
- get_home_directory(): Retrieves and returns the home directory path for the current user based on the operating system.
- print_time(): Given start and end times in seconds , returns human-readable formatted string representing elapsed time