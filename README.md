# Building Segmentation with SAM & PyTorch (DFC19)

A deep learning project focused on semantic segmentation of buildings from aerial imagery. The project uses the Data Fusion Contest 2019 (DFC19) dataset, featuring building mask annotations generated/labeled using Meta's Segment Anything Model (SAM).

## Features

- **Training Pipeline**: Trains a PyTorch semantic segmentation model (like UNet/FPN) using the `segmentation_models_pytorch` (smp) library.
- **SAM Mask Processing**: Converts Segment Anything Model (SAM) JSON outputs or binary masks into georeferenced TIFF format (`sam_mask_to_tif.py`).
- **Data Preprocessing & Augmentation**: Employs `albumentations` for heavy image augmentations to improve generalization.
- **Inference & Prediction**: Runs predictions on test images and produces segmentation masks.

## Project Structure

- `train_dfc19.py`: Defines the Dataset class, loads imagery, configures the segmentation model, and runs the training loop.
- `predict_dfc19.py`: Loads a trained model weight to perform segmentation inference on query images.
- `sam_mask_to_tif.py`: Converts raw mask predictions or annotations into geospatial TIFF files.
- `convert-masks-to-json.py`: Formatting utility to convert segmentations into JSON structures.
- `embed_image_files.py`: Utility to handle image embedding generation.
- `check.py`: Validation script to sanity-check data alignment (images vs. masks).

## Setup and Usage

### Prerequisites
Install PyTorch and computer vision packages:
```bash
pip install torch torchvision segmentation-models-pytorch albumentations opencv-python numpy tqdm
```

### Training
To train the building segmentation model:
```bash
python train_dfc19.py
```

### Inference
To make predictions using trained weights:
```bash
python predict_dfc19.py
```
