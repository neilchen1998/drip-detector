from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import shutil
import os

def prepare_data_dir(data_dir: Path):
    '''Prepare

    Parameters:
        data_dir: the path of the dataset
    '''

    # Create the main directory
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    for split in ['train', 'val', 'test']:
        (data_dir/'images'/split).mkdir(parents=True, exist_ok=True)
        (data_dir/'labels'/split).mkdir(parents=True, exist_ok=True)

    return

def download_dataset(path: str):
    '''Download the dataset from Hugging Face

    Parameters:
        path: the source of the dataset
    '''

    try:
        print(f"Downloading dataset from: {path}...\n")
        dataset = load_dataset(path)
        return dataset
    except Exception as e:
        print(f"Error: {e}")
        return

def convert_to_yolo_format(dataset, dir: Path, split: str, N = -1):
    '''Download the dataset from Hugging Face

    Parameters:
        dataset: the dataset
        dir: the directory of the dataset
        split: the name of the split (train, validation, etc.)
        N: the number of images (-1 will include all images)
    '''

    # Construct the filepaths
    img_dir = dir/'images'/split
    label_dir = dir/'labels'/split

    # Set the desired number of images
    if N == -1:
        N = len(dataset[split])
    else:
        N = min(N, len(dataset[split]))

    # Convert the data to YOLO format
    for i, example in enumerate(tqdm(dataset[split], desc=f"Processing {split} data")):

        # Image extraction
        img = example['image']
        img_width, img_height = img.size

        # Save images
        img_filename = f"{example['image_id']}.jpg"
        img.save(img_dir/img_filename)

        # Convert annotations
        label_lines = []
        objects = example['objects']

        # Iterate all objects that are detected
        for category_id, bbox in zip(objects['category'], objects['bbox']):

            # Get the metadata of the bounding box
            x_min, y_min, x_max, y_max = bbox

            # Convert to YOLO format
            # height of the box
            # width of the box
            # the center of x
            # the center of y
            box_width = x_max - x_min
            box_height = y_max - y_min
            x_center = x_min + box_width / 2
            y_center = y_min + box_height / 2

            # Normalized the values
            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            box_width_norm = box_width / img_width
            box_height_norm = box_height / img_height

            # Append the values to the list
            label_lines.append(f"{category_id} {x_center_norm:.6f} {y_center_norm:.6f} {box_width_norm:.6f} {box_height_norm:.6f}")

        label_filename = f"{example['image_id']}.txt"
        (label_dir/label_filename).write_text('\n'.join(label_lines))

        if i >= (N - 1):
            break

    print(f"Processed {N} images in {split} split.")

def prepare_data_pipeline(dir: str, N = 200):
    '''Download the dataset and convert it to YOLO's format

    Parameters:
        dir: the source of YOLO dataset
        N: the number of samples
    '''

    YOLO_DATA_DIR = 'fashionpedia_yolo'

    # Check if the directory exists
    if os.path.exists(YOLO_DATA_DIR):
        shutil.rmtree(YOLO_DATA_DIR)
        print(f"Removed {YOLO_DATA_DIR}.\n")

    dataset = download_dataset(dir)
    prepare_data_dir(Path(YOLO_DATA_DIR))
    convert_to_yolo_format(dataset, Path(YOLO_DATA_DIR), 'train', N)
    convert_to_yolo_format(dataset, Path(YOLO_DATA_DIR), 'val', N)

    return
