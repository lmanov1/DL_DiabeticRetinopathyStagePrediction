import os

def is_nvidia_gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

if is_nvidia_gpu_available():
    os.system("poetry add torch torchvision torchaudio")
else:
    os.system("poetry add torch-cpu torchvision-cpu torchaudio-cpu")
