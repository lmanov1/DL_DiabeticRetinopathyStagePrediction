# This will be run manually by the user to check for GPU availability and install CUDA if necessary
# from Util import check_hardware_and_install as check_hardware_and_install
# check_hardware_and_install.install_dependencies()
import subprocess

# Import necessary modules
import torch
import torch.nn as nn
import torch.optim as optim
from fastai.vision.all import *
from Util.MultiPlatform import *
import time

from code.models.train_model import PretrainedEyeDiseaseClassifier, EyeDiseaseClassifier

# Define the dataset names and paths
# Choose here the dataset(s) you want to download (https://www.kaggle.com/search?q=APTOS+2019+Blindness+Detection+Dataset+in%3Adatasets)
# Dataset names can be found under the  https://www.kaggle.com/datasets page
DATASET_PATH = standardize_path('data/raw')
DATASET_NAME_resized15_19 = standardize_path('benjaminwarner/resized-2015-2019-blindness-detection-images')
DATASET_NAME_aptos19 = standardize_path('mariaherrerot/aptos2019')

# Define the dataset structure for validation and test datasets
dataset_train_structure_resized15_19 = [
    {
        'labels': construct_path(DATASET_NAME_resized15_19, 'labels', 'trainLabels15.csv'),
        'images': construct_path(DATASET_NAME_resized15_19, 'resized train 15')
    },
    {
        'labels': construct_path(DATASET_NAME_resized15_19, 'labels', 'trainLabels19.csv'),
        'images': construct_path(DATASET_NAME_resized15_19, 'resized train 19')
    }
]

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


# Define your main function
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

    pretrained_model = PretrainedEyeDiseaseClassifier(num_classes=num_dr_classes, pretrained_model=pretrained_models[0])
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLossFlat
    # pretr_optimizer = optim.Adam(pretrained_model.parameters(), lr=0.001)

    inf_model = EyeDiseaseClassifier(num_classes=num_dr_classes)
    # inf_optimizer = optim.Adam(inf_model.parameters(), lr=0.001)

    inf_model.to(device)
    pretrained_model.to(device)

    for key, dls in train_dataloaders.items():
        #1. Train the (pretrained) model
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
            print_time(start_time, end_time, "Pretrained model training time")
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

        print_time(start_time, end_time, "CNN model training time")
        # Save the trained model weights
        trained_model_file_name = dataset_name + '_trained_model.pth'
        trained_weigths_path = os.path.join(os.getcwd(), 'data', 'output', trained_model_file_name).replace('/', get_path_separator())        
        print(" ===> Saving train model to ", trained_weigths_path)
        torch.save(inf_model.state_dict(), trained_weigths_path )
        inf_model.evaluate_model(dls)

# Call the main function
if __name__ == "__main__":
    main()
