# Face Detection with TensorFlow and OpenCV

A deep learning project that detects a face and predicts its bounding box in images and webcam frames.

The project covers the complete workflow: collecting images, manually annotating faces, preparing and augmenting data, training a model, and testing predictions.

## Technologies

- Python
- TensorFlow / Keras
- OpenCV
- Albumentations
- NumPy
- Matplotlib
- LabelMe
- Jupyter Notebook

## Model Architecture

The model uses a pretrained **VGG16 backbone** with two output branches:

| Branch | Purpose | Output |
| --- | --- | --- |
| Classification | Predict whether a face is present | Face confidence score |
| Bounding-box regression | Predict the face location | Four normalized coordinates |

Each branch uses global max pooling followed by a dense layer with 2,048 ReLU units and a sigmoid output layer.

Images are resized to **120 × 120 pixels**, converted to RGB, and scaled to `[0, 1]`.

The training objective combines binary classification loss and localization loss:

```text
Total loss = Localization loss + 0.5 × Classification loss
```

The current implementation predicts **one bounding box per image**. It does not identify people or perform temporal tracking.

## Project Workflow

1. Collect face and no-face images using a webcam.
2. Annotate face images with bounding boxes using LabelMe.
3. Split original images into training, validation, and test sets.
4. Apply image transformations and update the corresponding bounding boxes.
5. Train the classification and localization branches.
6. Inspect predictions on test images.
7. Run inference on webcam frames.

## Development and Improvements

The initial implementation followed a tutorial. Further work focused on investigating problems observed during live testing.

### Correcting bounding-box scaling

The original drawing code used fixed dimensions of `450 × 450`, even when displaying a larger webcam frame.

The updated code scales normalized coordinates using the actual frame width and height.

### Expanding the dataset

Initial images mostly contained close-up faces. Additional images were collected at different distances and positions to improve the variety of training examples.

### Preserving faces near image boundaries

Random cropping sometimes removed faces located near the edges of the original image.

Full-image resizing replaced random cropping to preserve the complete scene and make training preprocessing more consistent with webcam inference. Resizing rectangular images to squares introduces some distortion, which remains a trade-off.

These changes address observed issues; their effect on generalization still needs quantitative evaluation.

## Repository Contents

| File | Description |
| --- | --- |
| `Face Detection.ipynb` | Data preparation, training, prediction, and webcam inference |
| `.gitignore` | Excludes local environments, private datasets, and generated files |
| `README.md` | Project documentation |

## Getting Started

### Requirements

Use a Python environment with:

- TensorFlow
- OpenCV
- Albumentations
- NumPy
- Matplotlib
- LabelMe
- Jupyter and IPython support

A pinned dependency file is not yet included. Review the notebook's installation cells before running them in an existing environment.

### Prepare your dataset

The personal dataset and trained model are not included. To run the complete workflow, collect and annotate your own images.

Use a LabelMe rectangle named `face` for face annotations.

The notebook expects this local folder structure:

```text
data/
├── images/
├── labels/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/

aug_data/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Create the necessary directories before generating augmented data.

Split original images **before augmentation**, and keep closely related captures in the same split to reduce data leakage.

The current annotation loader treats missing annotation files as no-face examples. Make sure missing labels are intentional. JSON files containing an empty `shapes` list require explicit handling.

### Train and run inference

1. Open `Face Detection.ipynb`.
2. Update example image filenames and camera settings for your environment.
3. Check that bounding-box normalization matches the original image dimensions.
4. Run preprocessing and inspect image–label alignment.
5. Build and train the model.
6. Save the trained model as a `.keras` file.
7. Use the same filename in the model-loading cell.
8. Run the webcam inference section.

Training the full VGG16 network can be slow on CPU.

## Evaluation

The current workflow includes:

- Training and validation loss plots
- Visual inspection of test predictions
- Manual testing with a webcam

No verified aggregate accuracy, precision, recall, IoU, or FPS benchmark is reported yet.

A high face confidence score indicates face presence; it does not guarantee an accurate bounding box.

## Limitations

- Predicts only one face box per image.
- Uses a small, personally collected dataset.
- Generalization to unfamiliar people, backgrounds, and lighting is not established.
- Full-image resizing changes the original aspect ratio.
- Custom training steps need explicit epoch-average loss tracking.
- Localization loss currently includes no-face examples.
- Validation and test augmentation should be replaced with a fixed evaluation protocol for reliable comparisons.

## Future Improvements

- Track average training and validation losses per epoch.
- Save checkpoints based on validation performance.
- Mask localization loss for no-face examples.
- Measure classification precision/recall and bounding-box IoU.
- Compare preprocessing approaches on the same held-out images.
- Expand the dataset across people, environments, and capture sessions.
- Separate training and inference into reusable scripts.
- Add a tested dependency file and automated notebook-output checks.
