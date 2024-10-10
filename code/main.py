import subprocess
# Import necessary modules
import torch

# Import your model class# Import your model class
from models import interface_model as Neural_Net
from data import data_preparation as data_preparation
from data import data_preprocessing as data_preprocessing


# Run the hardware detection and installation script
subprocess.run(['python', 'Util/check_hardware_and_install.py'], check=True)

# Check the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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
