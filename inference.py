from ultralytics import YOLO
from pathlib import Path

def inference(pic_path: str, weights_path: str):
    """Inferences the picture with the pretrained model

    Parameter:
        pic_path: the file path of the picture
        weights_path: the file path of the weights
    """

    # Check if the picture exists
    if Path(pic_path).exists() == False:
        print(f"Error: {pic_path} does not exist.\n")
        return

    # Check if the weight file exists
    if Path(weights_path).exists() == False:
        print(f"Error: {weights_path} does not exist.\n")
        return

    # Load the weights and predict the given picture
    model = YOLO(weights_path)
    results = model.predict(pic_path)

    # Print the number of bounding boxes
    if results[0].boxes:
        print(f"Number of detected boxes: {len(results[0].boxes)}")
    else:
        print("No objects detected in this image.")

    # Show the annotated picture
    results[0].show()

    return

if __name__ == '__main__':

    pic_path = "pic_1.jpg"
    weights_path = "runs/detect/yolov8n_fashionpedia14/weights/best.pt"

    inference(pic_path, weights_path)