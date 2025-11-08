import torch

def get_device_type():
    '''Get the device type'''

    # Check for CUDA GPU
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        print("CUDA GPU detected.")
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'
        print("Apple Silicon GPU detected.")
    else:
        DEVICE = 'cpu'
        print("WARNING: No GPU detected. Using CPU.")

    return DEVICE
