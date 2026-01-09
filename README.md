This repository provides an implementation of a YOLOv10-based object detection framework for robust rebar section detection in complex construction site environments. The proposed method is designed for UAV-acquired images and supports accurate rebar localization to facilitate subsequent geometric measurement tasks.

Overview
YOLOv10 is adopted as the baseline object detection network due to its favorable balance between detection accuracy and inference efficiency. Based on YOLOv10, this project integrates task-specific modifications (e.g., image enhancement modules) to improve robustness under challenging conditions such as uneven illumination, motion blur, and background clutter commonly observed on construction sites.

The framework supports end-to-end training and inference using YAML-based configuration files, following the standard practice of modern YOLO implementations.

Environment Setup
The code is developed and tested under the following environment:

Python ≥ 3.8

PyTorch ≥ 2.0

CUDA ≥ 11.7 (recommended for GPU acceleration)

Ultralytics YOLO framework (YOLOv10-compatible version)

Install the required dependencies using:

pip install -r requirements.txt

Project Structure . ├── models/ # YOLOv10 model configuration files (.yaml) ├── data/ # Dataset configuration files │ └── data.yaml ├── cfg/ │ └── hyp.yaml # Training hyperparameter configuration ├── train.py # Training script ├── val.py # Validation script ├── detect.py # Inference script ├── data.txt # Dataset download link └── README.md

Dataset Configuration

The dataset paths and class definitions are specified in a YAML file (e.g., data.yaml):

train: path/to/train/images val: path/to/val/images test: path/to/test/images

nc: 1 names: ['rebar']

The dataset includes UAV-acquired images collected under real construction site conditions and is used to evaluate the generalization performance of the detector.

Training
Model training is performed using YAML-based configuration files combined with command-line arguments. An example training command is shown below:

yolo detect train
model=models/yolov10.yaml
data=data/data.yaml
epochs=200
batch=16
imgsz=640
optimizer=AdamW

Training hyperparameters such as learning rate, momentum, and weight decay are defined in hyp.yaml and can be adjusted as needed.

Inference
After training, the detector can be applied to new images using:

yolo detect predict
model=best.pt
source=path/to/images

The output includes bounding boxes and confidence scores for detected rebar sections.

Notes on Data Availability
The dataset used in this project is publicly available for research purposes. The download link and related information are provided in data.txt.

Please refer to data.txt for detailed instructions on how to access and use the dataset.