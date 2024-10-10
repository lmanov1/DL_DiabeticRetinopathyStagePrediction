import os
import subprocess
import sys

def is_nvidia_gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def run_command(command):
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        sys.exit(f"Command failed: {command}")

def remove_if_installed(package):
    result = subprocess.run(f"pip show {package}", shell=True)
    if result.returncode == 0:
        print(f"Removing {package}...")
        run_command(f"pip uninstall -y {package}")

if is_nvidia_gpu_available():
    print("Nvidia GPU found. Installing CUDA-supported PyTorch...")
    run_command("poetry add torch torchvision torchaudio tensorflow==2.17.0")
else:
    print("No Nvidia GPU found. Installing CPU-only PyTorch...")
    for package in ["torch", "torchvision", "torchaudio", "tensorflow"]:
        remove_if_installed(package)
    run_command("pip install torch==2.0.0+cpu torchvision==0.15.1+cpu torchaudio==2.0.0+cpu tensorflow-cpu==2.17.0 -f https://download.pytorch.org/whl/cpu/torch_stable.html")
    run_command("pip install numpy==1.23.5")  # Ensure compatible numpy version

print("Setup complete.")
