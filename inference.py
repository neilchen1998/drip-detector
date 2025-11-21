from ultralytics import YOLO   
from pathlib import Path	
import argparse

from utils.helper import print_info

# The default values
CONFIDENCE_SCORE = 0.6
TARGET_DIRECTORY = "inference/"
WEIGHTS_FILEPATH = "runs/detect/yolov8n_fashionpedia14/weights/best.pt"

def check_image_directory(pic_dir_path: str):
    """Checks if the given picture directory exists and contains at least one image

    Parameter:
        pic_dir_path: the directory of the picture(s)
    """

    pic_dir = Path(pic_dir_path)

    # Check if the parameter is a directory and if it exists
    if not pic_dir.is_dir() or not pic_dir.exists():
        print(f"Error: {pic_dir_path} does not exist.\n")
        return False

    # Check if there is at least one image in the directory
    if not any(f.suffix.lower() in ['.jpg', '.jpeg', '.png'] for f in pic_dir.iterdir() if f.is_file()):
        print(f"Error: {pic_dir_path} does not contain at least one image.\n")
        return False

    return True

def inference(weights_path: str, pic_dir_path: str, conf=CONFIDENCE_SCORE):
    """Inferences all pictures in the given directory with the given weights

    Parameter:
        weights_path: the filepath of the weights
        pic_dir_path: the directory of the picture(s)
        conf: the confidence score (must be between 0.0 and 1.0)
    """

    # Check if there is at least one image in the directory
    if not check_image_directory(pic_dir_path):
        return

    # Check if the weight file exists
    if not Path(weights_path).exists():
        print(f"Error: {weights_path} does not exist.\n")
        return

    if conf > 1.0 or conf < 0.0:
        print(f"Error: confidence score must be between 0.0 and 1.0.\n")
        return

    # Load the model
    model = YOLO(weights_path)

    # Predict the images in batches
    results = model.predict(Path(pic_dir_path), conf=conf)

    for result in results:

        # Retreive the filename
        img_path = result.path
        filename = Path(img_path).name

        # Only show images that there is at least one hit
        if result.boxes and len(result.boxes) > 0:
            print(f"{len(result.boxes)} object(s) detected in {filename}.")
            result.show()
        else:
            print(f"No objects detected in {filename}.")

    return

if __name__ == '__main__':

    # Parse the user's arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-w',
        '--weights_filepath',
        type=str,
        default=WEIGHTS_FILEPATH,
        help="The filepath of the weights",
    )
    parser.add_argument(
        '-t',
        '--target_directory',
        type=str,
        help="The target directory.",
        metavar="TARGET_DIRECTORY",
        default=TARGET_DIRECTORY,
    )
    parser.add_argument(
        '-c',
        '--conf',
        type=float,
        default=CONFIDENCE_SCORE,
        help=f"The confidence score (default: {CONFIDENCE_SCORE})",
    )
    args = parser.parse_args()

    # Access and print the argument values
    weights_filepath = args.weights_filepath
    target_directory = args.target_directory
    conf = args.conf

    # Construct the header
    title = "Inference"
    lines = [
        f"The filepath of the weights: {weights_filepath}",
        f"The target directory: {target_directory}",
        f"The confidence score: {conf}"
    ]

    print_info(title, lines)

    inference(weights_filepath, target_directory, conf)
