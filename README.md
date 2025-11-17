# Drip Detector

## Introduction

The main goal of this project is to develop a robust solution for clothing piece detection-such as shirts, trousers, jackets, etc-across a myriad of sources, including but not limited to, professional magazine editorials, online appearal listings, online influencer posts.

## Installation

To install all the packages, run the following comamnds with conda or pip:

```zsh
pip install ultralytics
pip install torch
pip install datasets
pip install tqdm
```

## Training

To train the model using the default number of images (600), run:

```zsh
python3 ./training.py
```

You can add an additional argument that specify the number of images that you would like to use for training:

```zsh
python3 ./training.py --sz <number_of_iamges>
```

## Inference

To use the model with pre-trained weights, run:

```zsh
python3 ./inference.py
```

If the weights or the images that you want to inference is stored somewhere else, run:

```zsh
python3 ./inference.py --weights <weights_filepath> --target <target_directory>
```

## Packages

### ultralytics

In this project, we use the **YOLO v8 nano** model created by **ultralytics**.
It is a compact version of the **YOLO v8** computer vision model, making an ideal candidate for light weight application.

### tqdm

[tqdm](https://github.com/tqdm/tqdm) is a fast progress bar tool in CLI written in Python.
It shows progress bars and the overhead is low.

### Apple Silicon Acceleration (For Mac Users Only)

In order to utilize the GPU on Mac, install the following packages with **conda**:

```zsh
conda install pytorch torchvision torchaudio -c pytorch-nightly
```

You can verify Metal Performance Shader (**mps**) is supported by running this script:

```python
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")
```

The output should show something like this:

```zsh
tensor([1.], device='mps:0')
```

# Dataset

The training dataset used in this project is **Fashionpedia** from **Hugging Face**.
It is consist of more than 45k images and 342k bounding boxes.
There are 46 categories of fashion pieces, such as shirt, watch, shoes, scarf, etc.
A raw bounding box from the dataset is defined as the four vertices of the box, which does not align with YOLO's format
Therefore, we need to convert the coordinates into what we need, i.e., the coordinate of the center of a bounding box, the height of a bounding box, and the width of a bounding box.

# You-Only-See-Once (YOLO)

YOLO is a computer vision model developed by Joseph Redmon et al. back in 2015.
It uses bounding boxes to locate and recognize objects.
The benefit is that it only requires a single pass.
The model outputs the center and the category of an object and it can detect multiple objects within a single input image.

## Reference

* [ultralytics](https://github.com/ultralytics/ultralytics)

* [tqdm](https://github.com/tqdm/tqdm)

* [Hugging Face](https://huggingface.co/)

* [Fashionpedia](https://huggingface.co/datasets/detection-datasets/fashionpedia)

* [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/)
