import torch
from pathlib import Path
from ultralytics import YOLO

from utils.data.downloader import prepare_data_pipeline
from utils.torch.info import get_device_type
from utils.ultralytics.config import create_yolo_yaml

# the path of fashionpedia dataset on 🤗 Hugging Face
HF_DATASET_PATH = "detection-datasets/fashionpedia"

# Model parameters
MODEL_NAME = 'yolov8n.pt'

# Training parameters
EPOCHS = 30
IMG_SIZE = 640
BATCH_SIZE = 16

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
        print("Dataset is already ready.\n")
        user_input = input("Redownload dataset? (y/n): ").lower()
        print("\n")

    # Check if dataset already exists and the user do not want to re-download the dataset
    if data_exists and user_input == 'n':
        print("User does not want to redownload.\n")
    else:
        prepare_data_pipeline(HF_DATASET_PATH, 100)

    config_path = create_yolo_yaml(YOLO_DATA_DIR, FASHIONPEDIA_CLASSES)

    try:
        # Load a pretrained YOLO model by model name
        model = YOLO(MODEL_NAME)

        # Train the model
        result = model.train(
            data=config_path,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            name='yolov8n_fashionpedia',
            device=get_device_type(),
            plots=False
        )
    except Exception as e:
        print(f"Error: {e}\n")
