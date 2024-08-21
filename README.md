# NASA Spacecraft Detection

## Overview

This project is part of the NASA Spacecraft Detection Challenge, aimed at developing new methods for conducting spacecraft inspections by identifying the boundaries of target spacecraft in images. 

The project includes data preprocessing, model training using YOLOv8, model evaluation, and a Streamlit web application for interactive spacecraft detection. The trained model can identify spacecraft in various images, including those with different backgrounds and potential image distortions.

https://www.drivendata.org/competitions/260/spacecraft-detection/page/832/

## Installation


## Usage

https://nasaspacecraftdetection-mvfmxebuyxmcpcyph4i9bg.streamlit.app/

## Data

20,000+ images and labels
80-10-10 split

## Model Training

YOLOv8 model 

## Evaluation

The model's performance is evaluated using the Jaccard Index, also known as the Intersection over Union (IoU). This metric is particularly well-suited for object detection tasks, as it measures the overlap between the predicted bounding box and the ground truth bounding box.
Jaccard Index (IoU) Explanation
The Jaccard Index is defined as the size of the intersection divided by the size of the union of two pixel sets:

J(A,B) = |A ∩ B| / |A ∪ B| = |A ∩ B| / (|A| + |B| - |A ∩ B|)

Where:

A is the set of pixels in the ground truth bounding box
B is the set of pixels in the predicted bounding box

### The Jaccard Index ranges from 0 to 1, where:

0 indicates no overlap between the predicted and ground truth bounding boxes
1 indicates perfect overlap (prediction exactly matches the ground truth)

A higher Jaccard Index indicates better performance, as it means the model's predictions more closely align with the true spacecraft locations in the images.

### Model Performance
Our trained model achieved an average Jaccard Index of 0.8703 across the test set.
This score indicates excellent performance:

It suggests that, on average, there is an 87.03% overlap between our model's predicted bounding boxes and the ground truth bounding boxes.
This high level of accuracy demonstrates that our model is highly effective at localizing spacecraft within images.
The score is particularly impressive given the challenges of the task, including varying spacecraft types, different backgrounds, and potential image distortions.


