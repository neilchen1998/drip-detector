import torch

from utils.data.downloader import download_dataset
from utils.torch.info import get_device_type

# the path of fashionpedia dataset on 🤗 Hugging Face
HF_DATASET_PATH = "detection-datasets/fashionpedia"

if __name__ == '__main__':

    _ = download_dataset(HF_DATASET_PATH)
