import torch

def check_mps_support():
    '''Check if MPS is found'''

    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print (x)
    else:
        print ("WARNING: MPS device not found.")

if __name__ == '__main__':

    check_mps_support()
