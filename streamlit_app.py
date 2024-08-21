import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from ultralytics import YOLO
import io

st.write('# 🚀 Spacecraft Detector 🛰️')

@st.cache_resource
def load_model():
    model_path = 'yolo_model/weights/best.pt'
    model = YOLO(model_path)
    return model

model = load_model()

uploaded_files = st.file_uploader("Choose up to 10 spacecraft images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files[:10]:  # Limit to 10 files
        # Read the file directly into a numpy array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        im_array = cv2.imdecode(file_bytes, 1)
        
        # Run model on the uploaded image
        results = model(im_array)
        
        for r in results:
            # Draw bounding boxes without labels
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cv2.rectangle(im_array, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Convert BGR to RGB
        im = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
        
        # Display the image using Streamlit
        st.image(im, caption=f"Processed: {uploaded_file.name}", use_column_width=True)
