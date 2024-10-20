# DL_DiabeticRetinopathyStagePrediction
This repo hosts a final DL project conducted as a part of data scientist certification at BIU



## Getting started

Tested with:     
python 3.10.12 (must be active version on your system) - if you have another python version , use `poetry update`     instead of `poetry install` during the step 5 - and does not commit poetry.lock afterwards!     
poetry 1.8.3    
pip 24.2    

1. Pre-requirements: 
    - Be sure you have poetry installed in your environment    
    - If you have GPU card (NVidia), be sure you have nvidia-smi cli installed         
2. clone the repo  , go to the cloned directory   
    `git clone git@github.com:lmanov1/DL_DiabeticRetinopathyStagePrediction.git`
3. If you have GPU on your system : run CUDA setup: this should detect GPU and install supporting python system libraries (unsupported by poetry) like CUDA     
    `python3 code/Util/check_hardware_and_install.py`    
4. Run `poetry update` (just once) - this will not use poetry.lock of the workspace but will rewrite it. Don't commit your poetry.lock        
    Don't worry , without available GPU (and CUDA) , tensorflow, torch and rest of libraries leveraging GPU will automatically use the CPU.     
5. Run `poetry shell`

6.  About Kaggle API  
    We use Kaggle API to download datasets from Kaggle.           
    To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com     
    Then go to the 'Account' tab of your user profile (https://www.kaggle.com/<username>/account) and select 'Create API Token'. This will trigger the download of kaggle.json, a file containing your API credentials. Place this file in the location appropriate for your operating system:

    - Linux: `$XDG_CONFIG_HOME/kaggle/kaggle.json` (defaults to `~/.config/kaggle/kaggle.json`). The path `~/.kaggle/kaggle.json` which was used by older versions of the tool is also still supported.
     `chmod 600 ~/.config/kaggle/kaggle.json` - no read access for other users.    
    - Windows: C:\Users\<Windows-username>\.kaggle\kaggle.json - you can check the exact location, sans drive, with echo %HOMEPATH%.        
    - Other: ~/.kaggle/kaggle.json     

    - You can define a shell environment variable KAGGLE_CONFIG_DIR to change this location to $KAGGLE_CONFIG_DIR/kaggle.json (on Windows it will be %KAGGLE_CONFIG_DIR%\kaggle.json).
    
    - You can also choose to export your Kaggle username and token to the environment:
    export KAGGLE_USERNAME=datadinosaur
    export KAGGLE_KEY=xxxxxxxxxxxxxx
    In addition, you can export any other configuration value that normally would be in the kaggle.json in the format 'KAGGLE_' (note uppercase).

7. Now you all set and can run project logics , for example       
        
    `python3 /code/main.py`  Running with `poetry run python.exe /code/main.py` on Windows does problems with import fastai , so use just `python.exe /code/main.py` Don't ask why.


## Project Blueprint: Diabetic Retinopathy Severity Classification
### Project Setup
    Dev Environment Setup:
    Determine available hardware (GPU) by manually running check_hardware_and_install.py from the code/Util folder. This will install CUDA where suitable. 
    Configure a virtual environment for package management. Please pay attention, both torch and TensorFlow do not maintain separate packages that depend upon underlying hardware( f.e. tensorflow-cpu and tesorflow-gpu)  for a while. Starting tensorflow 2.17
    There are some warnings in run time - that can be ignored.
    Existing code should support both CPU/GPU environments, Windows, Linux, and (maybe 🙂) MAC
### Data Handling
#### Data Acquisition:
    Download datasets from Kaggle: 2015 Diabetic Retinopathy Detection, APTOS 2019 Blindness Detection.
    Organize the datasets in a structured format.
#### Data Preprocessing:
    Clean and normalize images.
    Implement data augmentation techniques.
    Split data into training, validation, and test sets.     

### Model Definition
#### Pretrained Model Setup      
    Define the PretrainedEyeDiseaseClassifier class with a pre-trained model (e.g., VGG16).
    Modify the classifier layers to match the number of classes.
#### Main Model Definition       
    Define the main EyeDiseaseClassifier class using a custom CNN or another architecture
        
### Training Process       
    Initial Training with Pretrained Model:
    Load data and create data loaders.
    Initialize the pre-trained model (e.g., VGG16).
    Train the pre-trained model with the available dataset.
    Save the pre-trained model’s weights.
    Main (inference) Model Training:
    Load the pre-trained model’s weights into the main model.
    Train the main EyeDiseaseClassifier model using the pre-trained weights for better performance.
    Alternatively, the main model can be trained from scratch  - to be decided
    Save the main model’s weights after training.
### Model Evaluation         
#### Performance Metrics        
    Evaluate both models on the test set.
    Calculate metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
### Deployment          
####  Model Saving     
    Save the trained models using appropriate file formats (.pth for PyTorch).
####  Inference Pipeline     
    Develop an inference pipeline for classifying new images.
###    API Development
    Hugging Face Deployment:
#### Deploy the trained model to Hugging Face with a graphical UI     
### Stretch Features         
#### LLM Assistant Integration      
    Integrate an LLM assistant to act as a virtual doctor.
    Implement functionalities for anamnesis, analysis, and providing recommendations.
#### Additional Disease Classification     
    Extend the model to classify other eye diseases like cataracts and glaucoma.
    Adjust the label set and retrain the model accordingly.
### Documentation      
#### Project Documentation    
    Maintain detailed documentation for each step.
    Include README files, code comments, and usage instructions.

### Detailed Code Description     
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



## Update requirements.txt
poetry export -f requirements.txt --output requirements.txt

## Production      
app.py  - main gradio application script.
Leverages existing        
update_production.py - upload relevant files from git to production space (`https://huggingface.co/spaces/Lmanov1/timm-efficientnet_b3.ra2_in1k`)           
For inference models persistant storage - we use a dataset repo (`https://huggingface.co/datasets/Lmanov1/BUI17_data`) 
To run  `update_production.py` there is a need to login to Hagging face with  a valid Hugging face API TOKEN. Currently token value being read from .env file where it should be stored in format:  MY_TOKEN="PUT HERE YOUR KEY".  This file is not managed by git , but local in the root directory of the project.
