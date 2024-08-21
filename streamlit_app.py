import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os

st.write('# 🚀 Spacecraft Detector 🛰️')

st.write("""
This app uses a YOLO (You Only Look Once) model to detect spacecraft in images. 
You can upload your own images of spacecraft, and the model will attempt to identify and highlight them.

Here are some tips for getting the best results:
1. Use clear, well-lit images
2. Ensure the spacecraft is the main focus of the image
3. The model works best with real photographs, not artistic renderings

Below is an example of how the detector works:
""")

@st.cache_resource
def load_model():
    model_path = 'yolo_model/weights/best.pt'  
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load the model: {str(e)}")
        return None

model = load_model()

if model is None:
    st.stop()

# Display example image with side-by-side comparison
example_image_path = '/artemis2.jpg'
if os.path.exists(example_image_path):
    original_image = cv2.imread(example_image_path)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Create a copy of the original image for drawing bounding boxes
    processed_image = original_image.copy()
    
    # Run model on the example image
    results = model(processed_image)
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cv2.rectangle(processed_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    
    # Display the original and processed images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image_rgb, caption="Original Image", use_column_width=True)
    with col2:
        st.image(processed_image_rgb, caption="Processed Image", use_column_width=True)
else:
    st.write("Example image not found. Please ensure 'artemis2.jpg' is in the 'assets' folder.")

st.write("Now, try uploading your own images!")

uploaded_files = st.file_uploader("Choose up to 10 spacecraft images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files[:10]:  # Limit to 10 files
        # Read the file directly into a numpy array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        im_array = cv2.imdecode(file_bytes, 1)
        
        # Run model on the uploaded image
        results = model(im_array)
        
        spacecraft_detected = False
        for r in results:
            if len(r.boxes) > 0:
                spacecraft_detected = True
                # Draw bounding boxes without labels
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cv2.rectangle(im_array, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Convert BGR to RGB
        im = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
        
        # Display the image using Streamlit
        st.image(im, caption=f"Processed: {uploaded_file.name}", use_column_width=True)
        
        # Display message if no spacecraft detected
        if not spacecraft_detected:
            st.write(f"Couldn't detect spacecraft in {uploaded_file.name}")
        else:
            st.write(f"Spacecraft detected in {uploaded_file.name}")
else:
    st.write("Upload an image to get started!")
