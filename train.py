import torch

from utils.data.downloader import prepare_data_pipeline
from utils.torch.info import get_device_type

# the path of fashionpedia dataset on 🤗 Hugging Face
HF_DATASET_PATH = "detection-datasets/fashionpedia"

if __name__ == '__main__':

    prepare_data_pipeline(HF_DATASET_PATH)
