import torch
from pathlib import Path

from utils.data.downloader import prepare_data_pipeline
from utils.torch.info import get_device_type
from utils.ultralytics.config import create_yolo_yaml

# the path of fashionpedia dataset on 🤗 Hugging Face
HF_DATASET_PATH = "detection-datasets/fashionpedia"

# the directory of YOLO
YOLO_DATA_DIR = Path('fashionpedia_yolo')

# the classes from fashionpedia
FASHIONPEDIA_CLASSES = [
    'shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants',
    'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat',
    'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt',
    'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, backpack, purse',
    'scarf', 'umbrella', 'wallet', 'sunglasses', 'trousers', 'suit', 'overcoat',
    'skirt, petticoat', 'jeans', 'cap', 'sneaker', 'heel', 'boot', 'sandal',
    'flat shoe', 'jewelry', 'bra', 'leggings', 'polo shirt', 'hoodie', 'belt bag'
]

if __name__ == '__main__':

    # Check if the required directory exists and at least one text file is present
    # In order to check if there is at least one text file inside the directory,
    # we must use glob function to iterate over directory and yeild existing files
    # i.e., create a generator
    # We then take one step and see if a file is found,
    # if a file is not found then it returns None which will make the bool variable to be False
    TRAIN_LABEL_DIR = YOLO_DATA_DIR / 'labels' / 'train'
    data_exists = TRAIN_LABEL_DIR.exists() and bool(next(TRAIN_LABEL_DIR.glob('*.txt'), None))

    if data_exists:
        print(f"Skipping download and conversion since labels exist in {TRAIN_LABEL_DIR}.\n")
    else:
        prepare_data_pipeline(HF_DATASET_PATH)

    config_path = create_yolo_yaml(YOLO_DATA_DIR, FASHIONPEDIA_CLASSES)
