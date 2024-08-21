import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import tempfile
import os
import requests
import io

# Install ultralytics if not already installed
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import ultralytics
except ImportError:
    install('ultralytics')
    import ultralytics

st.write('# Spacecraft Detector')

@st.cache_resource
def load_model():
    # GitHub raw content URL for the model weights
    url = "https://github.com/Cp557/nasa_spacecraft_detection/raw/main/yolo_model/weights/best.pt"
    response = requests.get(url)
    model_bytes = io.BytesIO(response.content)
    model = YOLO(model_bytes)
    return model

model = load_model()

uploaded_files = st.file_uploader("Choose up to 10 spacecraft images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files[:10]:  # Limit to 10 files
        # Create a temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Run model on the uploaded image
        results = model(tmp_file_path)
        
        for r in results:
            # Get the original image
            im_array = cv2.imread(tmp_file_path)
            
            # Draw bounding boxes without labels
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cv2.rectangle(im_array, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Convert BGR to RGB
            im = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
            
            # Display the image using Streamlit
            st.image(im, caption=f"Processed: {uploaded_file.name}", use_column_width=True)

        # Remove the temporary file
        os.unlink(tmp_file_path)
