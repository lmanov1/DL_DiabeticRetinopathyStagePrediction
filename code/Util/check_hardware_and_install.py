import os
import subprocess
import platform
import sys


def is_nvidia_gpu_available():
    #nvidia-smi is a command-line utility that comes with the NVIDIA GPU display drivers. It provides information about the GPU, including its utilization, temperature, and memory usage.
    # If the command is not found, it likely means that the NVIDIA drivers are not installed, and therefore, the system does not have an NVIDIA GPU.

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
             # Sorry Mac users, I don't have a Mac to test this on. Please let me know if this works.
            result = subprocess.run(["/usr/local/cuda/bin/nvcc", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else: # Linux and Windows
            result = subprocess.run(nvidia_smi_command_list, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)                

        # Pay attention , GPU/CUDA users should install nvidia-smi first                          
        gpu_count = int(result.stdout.decode().split('\n')[1].strip())
        print(f"Number of NVIDIA GPUs available: {gpu_count}")
        return gpu_count > 0

    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_dependencies():
    if is_nvidia_gpu_available():
        print("GPU detected. Installing CUDA...")
        if platform.system() == "Windows":
            os.system("winget install cuda")  # Assuming you use winget, adjust command as necessary
        elif platform.system() == "Darwin":  # macOS
            os.system("brew install cuda")
        else:  # Linux (Ubuntu, Fedora, Arch, etc.)
            # Determine the package manager type on current Linux
            package_manager = None
            if os.path.exists("/usr/bin/apt-get"):
                package_manager = "apt-get"
            elif os.path.exists("/usr/bin/yum"):
                package_manager = "yum"
            elif os.path.exists("/usr/bin/pacman"):
                package_manager = "pacman"
            else:
                raise EnvironmentError("Unsupported package manager")

            # Install CUDA using the determined package manager
            if package_manager == "apt-get":
                os.system("sudo apt-get update")
                os.system("sudo apt-get install -y cuda")
            elif package_manager == "yum":
                os.system("sudo yum update")
                os.system("sudo yum install -y cuda")
            elif package_manager == "pacman":
                os.system("sudo pacman -Syu")
                os.system("sudo pacman -S --noconfirm cuda")
            os.system("sudo apt-get install cuda")
        
    else:

        print("No GPU detected")
    
    # once CUDA is installed, you can install any cuda- leveraging libraries - use poetry install
    
        print("No GPU detected. Installing CPU-only TensorFlow...")
        tensorflow_version = "tensorflow==2.10.0"
        io_version = "tensorflow-io-gcs-filesystem==0.29.0"

    # Install the packages
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', tensorflow_version])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', io_version])

    #specific to WIndows
    if platform.system() == "Windows":
        print("Installing tqdm on Windows...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tqdm'])
        print("No GPU detected")
    
    # once CUDA is installed, you can install any cuda- leveraging libraries like
    # tensorflow, keras , torch - handled in pyproject.toml (poetry install)

def run_command(command):
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        sys.exit(f"Command failed: {command}")

def remove_if_installed(package):
    result = subprocess.run(f"pip show {package}", shell=True)
    if result.returncode == 0:
        print(f"Removing {package}...")
        run_command(f"pip uninstall -y {package}")

install_dependencies()
print("Setup complete.")
