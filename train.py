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

def setup_model(model_name: str):
    """Initializes YOLO model

    Parameter:
        model_name: the name of the model

    Return:
        the YOLO model
    """

    model = YOLO(model_name)

    return model

def train_mode(model, config_path: str):
    '''Train the model

    Parameters:
        model: the model
        config_path: the path of the YOLO YAML

    Returns:
        None: the result
    '''

    # Train the model
    result = model.train(
        data=config_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name='yolov8n_fashionpedia',
        device=get_device_type(),   # uses the GPU to train the model if available
        patience=10,
        save=True,
        pretrained=True,
        optimizer='Adam',
        verbose=True,
        plots=True,

        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
    )

    return result

def validate_model(model, config_path: str):
    '''Validate the model

    Parameters:
        model: the model
        config_path: the path of the YOLO YAML

    Returns:
        None: the result
    '''

    metrics = model.val(data=str(config_path))

    # Print the results
    print("Validation results:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    return metrics

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
        prepare_data_pipeline(HF_DATASET_PATH, 600)

    config_path = create_yolo_yaml(YOLO_DATA_DIR, FASHIONPEDIA_CLASSES)

    # Set up the model
    model = setup_model(MODEL_NAME)

    # Train the model
    train_mode(model, config_path)

    # Validate the model
    validate_model(model, config_path)
