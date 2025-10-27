# Vehicle-Detection-Yolo8-

A simple and practical implementation of vehicle detection using Ultralytics YOLOv8. This repository demonstrates how to prepare a custom dataset, train a YOLOv8 model, and run inference on images and videos to detect vehicles (cars, trucks, buses, motorcycles, etc.).

## Features

- Training with custom YOLO-format datasets
- Inference on images, directories of images, and video files
- Save and export predictions and annotated visualizations
- Example training, evaluation, and export commands using Ultralytics YOLOv8

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset structure](#dataset-structure)
- [Prepare your dataset](#prepare-your-dataset)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Model export](#model-export)
- [Directory layout](#directory-layout)
- [Notes and tips](#notes-and-tips)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Requirements

- Python 3.8+
- pip
- (Optional) NVIDIA GPU with CUDA for faster training

Recommended Python packages:
- ultralytics (YOLOv8)
- opencv-python
- matplotlib
- (Optional) tqdm, seaborn

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/AshraqtTamer/Vehicle-Detection-Yolo8-.git
cd Vehicle-Detection-Yolo8-
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

3. Upgrade pip and install dependencies:

```bash
pip install -U pip
pip install ultralytics opencv-python matplotlib
```

4. (Optional) If this repository has a requirements.txt, you can instead run:

```bash
pip install -r requirements.txt
```

---

## Dataset structure

This repo expects datasets formatted in YOLO format (image + label pairs). A typical structure:

```
dataset/
├─ images/
│  ├─ train/
│  ├─ val/
│  └─ test/
└─ labels/
   ├─ train/
   ├─ val/
   └─ test/
```

- Images: jpg, png, etc.
- Labels: plain text files with one line per object:
  <class_id> <x_center> <y_center> <width> <height> (normalized 0..1)

Create a dataset YAML file (example: data/vehicle.yaml):

```yaml
train: ../dataset/images/train
val:   ../dataset/images/val
test:  ../dataset/images/test

nc: 4
names: ['car', 'truck', 'bus', 'motorcycle']
```

Adjust paths and classes to match your dataset.

---

## Prepare your dataset

- Label images using tools like LabelImg (YOLO format), Roboflow export, CVAT, or Label Studio.
- Ensure every image has a corresponding .txt label file in the labels folder (same base filename).
- Keep class ordering consistent with the YAML config.
- Verify bounding boxes are normalized and within [0, 1].

---

## Training

A simple example training command using Ultralytics YOLOv8:

```bash
yolo task=detect mode=train model=yolov8n.pt data=data/vehicle.yaml epochs=50 imgsz=640 batch=16 name=vehicle_yolov8n
```

Key flags:
- model: starting checkpoint (e.g., yolov8n.pt, yolov8s.pt)
- data: YAML config for your dataset
- epochs: number of training epochs
- imgsz: input image size
- batch: batch size
- name: run name (results saved under runs/detect/{name})

Checkpoints and logs will be saved to runs/detect/{name} by default.

---

## Inference

Predict on a single image:

```bash
yolo task=detect mode=predict model=runs/detect/vehicle_yolov8n/weights/best.pt source=inference/images/car.jpg save=True
```

Predict on a directory or video:

```bash
# Directory of images
yolo task=detect mode=predict model=... source=inference/images/ save=True

# Video file
yolo task=detect mode=predict model=... source=inference/videos/traffic.mp4 save=True

# Webcam (device 0)
yolo task=detect mode=predict model=... source=0
```

- save=True saves annotated results (images/videos) and predictions.
- Use additional Ultralytics flags to control confidence threshold (conf), IoU (iou), etc.

---

## Evaluation

Evaluate a trained model on validation/test set:

```bash
yolo task=detect mode=val model=runs/detect/vehicle_yolov8n/weights/best.pt data=data/vehicle.yaml
```

This computes mAP and other detection metrics. You can pass flags to tailor thresholds or metrics.

---

## Model export

Export your trained model to formats like ONNX, TensorRT, CoreML:

```bash
yolo export model=runs/detect/vehicle_yolov8n/weights/best.pt format=onnx
```

Refer to Ultralytics documentation for additional formats and export options.

---

## Directory layout

Suggested repository layout:

- data/                 # dataset YAML configs
- dataset/              # images and labels (train/val/test)
- inference/            # sample images and videos for quick testing
- runs/                 # training and inference outputs (created after runs)
- models/               # exported models (onnx, tflite, etc.)
- notebooks/            # optional Jupyter notebooks
- README.md
- LICENSE

---

## Notes and tips

- Start with small models (yolov8n, yolov8s) for quick prototyping; switch to larger models (yolov8m, yolov8l) for better accuracy.
- If training is unstable:
  - Lower learning rate
  - Reduce batch size
  - Increase data augmentation
- Use a GPU and CUDA for practical training times.
- Consider using Roboflow or augmentation libraries to balance/expand datasets.
- Monitor training with Ultralytics logs and TensorBoard if desired.

---

## Contributing

Contributions, bug reports, and enhancements are welcome. To contribute:
1. Fork the repository
2. Create a branch for your change
3. Make your changes, add tests if applicable
4. Open a pull request describing your changes

Please follow code style and include clear commit messages.

---

## Acknowledgements

- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- Labeling tools and dataset providers such as LabelImg and Roboflow

---

## Contact

If you have questions or suggestions, open an issue on this repository or contact the repository owner.