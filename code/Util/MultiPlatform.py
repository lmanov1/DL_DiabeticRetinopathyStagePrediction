import os
import platform


def get_path_separator():
        return os.path.sep
        # system = platform.system()
        # if system == 'Linux' or system == 'Darwin':  # Darwin is macOS
        #     separator = '/'
        # elif system == 'Windows':
        #     separator = '\\'
        # else:
        #     raise EnvironmentError("Unsupported operating system")
        
        # print(f"Running on {system}. Path separator: {separator}")
        # return separator
    
def get_home_directory():    
    system = platform.system()
    if system == 'Linux' or system == 'Darwin':  # Darwin is macOS
        home_dir = os.environ.get('HOME')        
    elif system == 'Windows':
        home_dir = os.environ.get('USERPROFILE')
    else:
        raise EnvironmentError("Unsupported operating system")
    
    print(f"Running on {system}. Home directory: {home_dir}")
    return home_dir


def print_time(start_secs, end_secs, title=""):
    elapsed_time = end_secs - start_secs
    days = elapsed_time // (24 * 3600)
    elapsed_time = elapsed_time % (24 * 3600)
    hours = elapsed_time // 3600
    elapsed_time %= 3600
    minutes = elapsed_time // 60
    seconds = elapsed_time % 60
    print(f"==> {title}: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds")
    return days, hours, minutes, seconds

def get_saved_model_name(dataset_name, output_format, model_name_prefix=""):
    """Returns the name of the saved model file."""
    
    model_file_name = model_name_prefix + dataset_name + "." + output_format
    return os.path.join(os.getcwd(), "data", "output", model_file_name).replace(
        "/", get_path_separator()
    )
	
def check_need_to_train_model(dls_key, dataset_name, model_name):
    """
    Checks if a model needs to be trained or if a saved model already exists.
    Args:
        dls_key (str): The key for the data loaders, used to generate the dataset name.
        dataset_name (str): The name of the dataset being used.
        model_name (str): The name of the model being used.
    Returns:
        tuple: A tuple containing:
            - bool: True if the model needs to be trained, False otherwise.
            - str: The path to the pretrained model file.
            - str: The file extension of the pretrained model ("pkl" or "pth").
    """
    """Checks if a model needs to be trained or if a saved model already exists."""
    # 1. Fine tune the (pretrained) model
    print("Current directory:", os.getcwd())
    # dataset name + specific data set name from currently processed data loaders
    # i.e. resnet50_aptos19train_1.pkl/pth
    print("Dataset name:", dataset_name)
    dataset_name = str(dataset_name + os.path.basename(dls_key).split(".")[0])

    pretrained_pkl_path = get_saved_model_name(
        dataset_name, "pkl", str(model_name + "_")
    )
    pretrained_torch_path = get_saved_model_name(
        dataset_name, "pth", str(model_name + "_")
    )
    print(
        " \n ===>  Looking for pretrained model here ",
        pretrained_torch_path,
        pretrained_pkl_path,
    )  # 513 MB

    # Depends on dataset , but in general it will be a very heavy run -
    # about 8 hours on 100% GPU - lets not run it again if we have the model

    if not os.path.exists(pretrained_pkl_path) and not os.path.exists(pretrained_torch_path):
        print("Trained model not found - fine-tuning now...")
        # win fix - Ensure directory exists
        directory = os.path.dirname(pretrained_pkl_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return True, pretrained_torch_path, pretrained_pkl_path , None
    else:
        if os.path.exists(pretrained_pkl_path):
            print(" ===>  Trained model already exists at ", pretrained_pkl_path)
            return False, pretrained_torch_path, pretrained_pkl_path, "pkl"
        
        elif os.path.exists(pretrained_torch_path):
            print(" ===>  Trained model already exists at ", pretrained_torch_path)
            return False, pretrained_torch_path, pretrained_pkl_path , "pth"