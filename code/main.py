# This will be run manually by the user to check for GPU availability and install CUDA if necessary
# from Util import check_hardware_and_install as check_hardware_and_install
# check_hardware_and_install.install_dependencies()

# Import necessary modules
import torch
# Import your model class# Import your model class
from models import inference_model as Neural_Net
from data import data_preparation as data_preparation
from data import data_preprocessing as data_preprocessing
from data import Dataloader as Dataloader

# Define your main function
def main():

    # data = None
    # model = None
    # # Example data transfer
    # data = data_preparation.DataPreparation(device)
    # model = Neural_Net.to(device)
    # # Your main code logic here
    # print("Starting the main function...")
    # # Example model and data transfer
    # model_interface = Neural_Net.ModelInterface(model)
    # data_prepared = data_preparation.DataPreparation(data)
    #
    #

    # Add your model training/evaluation code
    print("Model and data are set up and ready!")


# Call the main function
if __name__ == "__main__":
    main()
