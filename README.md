# NASA Spacecraft Detection

## Overview
This project is part of the [NASA Spacecraft Detection Challenge](https://www.drivendata.org/competitions/260/spacecraft-detection/page/832/), aimed at developing new methods for conducting spacecraft inspections by identifying the boundaries of target spacecraft in images. 

The project includes data preprocessing, model training using YOLOv8, model evaluation, and a Streamlit web application for interactive spacecraft detection. The trained model can identify spacecraft with high accuracy.

## Project Structure
```
.
├── yolo_model/            # Directory containing model weights
├── README.md              # This file
├── artemis2.jpg           # Example image for the Streamlit app
├── data.yaml              # YAML file for data configuration
├── data_preprocessing.ipynb  # Jupyter notebook for data preprocessing
├── evaluation.ipynb       # Jupyter notebook for model evaluation
├── packages.txt           # List of system packages required
├── requirements.txt       # List of Python packages required
├── streamlit_app.py       # Streamlit application for interactive detection
└── yolo_training.ipynb    # Jupyter notebook for model training
```

## Usage
You can access the interactive spacecraft detection application at:
[https://nasaspacecraftdetection-mvfmxebuyxmcpcyph4i9bg.streamlit.app/](https://nasaspacecraftdetection-mvfmxebuyxmcpcyph4i9bg.streamlit.app/)

## Data and Preprocessing

### Dataset
The dataset consists of over 20,000 images and their corresponding labels. The data is split as follows:
- 80% Training set
- 10% Validation set
- 10% Test set

### Preprocessing Steps
The data preprocessing is detailed in the `data_preprocessing.ipynb` notebook. Key steps include:

1. **Data Loading**: 
   - Load training labels, metadata, and submission format from CSV files.

2. **Image Visualization**: 
   - Display sample images with bounding boxes for verification.

3. **Label Conversion**: 
   - Convert bounding box coordinates to YOLOv8 format.
   - YOLOv8 format: `<class> <x_center> <y_center> <width> <height>`
   - All values are normalized to be between 0 and 1.

4. **YOLO Label Creation**:
   - Create text files for each image containing the YOLO format labels.
   - Each line in the text file represents one object (spacecraft) in the image.

This preprocessing pipeline ensures that the data is in the correct format for training the YOLOv8 model and allows for proper evaluation of the model's performance.

## Model Training
The model used for this project is YOLOv8 (You Only Look Once version 8). The training process is in the `yolo_training.ipynb` notebook.

## Evaluation
The model's performance is evaluated using the Jaccard Index, also known as the Intersection over Union (IoU). This metric is particularly well-suited for object detection tasks, as it measures the overlap between the predicted bounding box and the ground truth bounding box.

### Jaccard Index (IoU) Explanation
The Jaccard Index is defined as the size of the intersection divided by the size of the union of two pixel sets:

J(A,B) = |A ∩ B| / |A ∪ B| = |A ∩ B| / (|A| + |B| - |A ∩ B|)

Where:
- A is the set of pixels in the ground truth bounding box
- B is the set of pixels in the predicted bounding box

The Jaccard Index ranges from 0 to 1, where:
- 0 indicates no overlap between the predicted and ground truth bounding boxes
- 1 indicates perfect overlap (prediction exactly matches the ground truth)

A higher Jaccard Index indicates better performance, as it means the model's predictions more closely align with the true spacecraft locations in the images.

### Model Performance
Our trained model achieved an average Jaccard Index of 0.8703 across the test set.

This score indicates excellent performance:
- It suggests that, on average, there is an 87.03% overlap between my model's predicted bounding boxes and the ground truth bounding boxes.
- This high level of accuracy demonstrates that our model is highly effective at localizing spacecraft within images.
- The score is particularly impressive given the challenges of the task, including varying spacecraft types, different backgrounds, and potential image distortions.

For more details on the evaluation process, refer to the `evaluation.ipynb` notebook.
