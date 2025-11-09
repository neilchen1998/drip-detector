import torch
from pathlib import Path
from ultralytics import YOLO
import yaml

def create_yolo_yaml(dir: Path, class_names: list):
    '''Generates the YAML configuration file for Ultralytics

    Parameters:
        dir: the path of the yaml file
        class_names: the list that contains all the name of the classes

    Returns:
        str: the path of the YAML file
    '''

    yaml_content = {
        'path': str(dir.resolve()), # make the path absolute
        'train': 'image/train',
        'val': 'image/val',
        'test': 'image/test',
        'nc': len(class_names), # the number of the classes
        'names': class_names,
    }

    # Create the YAML file
    yaml_path = dir/'fashionpedia.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print(f"Successfully created YAML file for YOLO at: {yaml_path.resolve()}.\n")

    return str(yaml_path)
