from ultralytics import YOLO
from pathlib import Path
import sys

def check_image_directory(pic_dir_path: str):
    """Checks if the given picture directory exists and contains at least one image

    Parameter:
        pic_dir_path: the directory of the picture
    """

    pic_dir = Path(pic_dir_path)

    if not pic_dir.exists() or not pic_dir.is_dir():
        print(f"Error: {pic_dir_path} does not exist.\n")
        return False

    if not any(f.suffix.lower() in ['.jpg', '.jpeg', '.png'] for f in pic_dir.iterdir() if f.is_file()):
        print(f"Error: {pic_dir_path} does not contain at least one image.\n")
        return False

    return True

def main():

    pic_dir_path = "inference/"
    weights_path = "runs/detect/yolov8n_fashionpedia13/weights/best.pt"

    # Check if the weight file exists
    if not check_image_directory(pic_dir_path):
        return

    # Check if the weight file exists
    if not Path(weights_path).exists():
        print(f"Error: {weights_path} does not exist.\n")
        return

    # Load the model
    model = YOLO(weights_path)

    # Predict the images in batches
    results = model.predict(Path(pic_dir_path), conf=0.4)

    for i, result in enumerate(results):

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

    main()
