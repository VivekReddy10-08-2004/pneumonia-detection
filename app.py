import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import densenet121
import os

# Set page config
st.set_page_config(
    page_title="Pneumonia Detector",
    layout="centered"
)

# Model configuration
MODEL_PATH = "densenet121_pneumonia_classifier.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing
def preprocess_image(image):
    """Preprocess image for DenseNet121"""
    # Standard ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image)

@st.cache_resource
def load_model():
    """Load the trained DenseNet121 model"""
    model = densenet121(pretrained=False)
    # Modify final layer for binary classification (Normal/Pneumonia)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    
    # Load trained weights
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    else:
        return None

def predict(image, model):
    """Make prediction on the image"""
    image_tensor = preprocess_image(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        confidence = probabilities[0][prediction[0]].item()
    
    return prediction.item(), confidence

# Load model once
model = load_model()

# Title and description
st.title("Pneumonia Detection System")
st.markdown("""
This application uses deep learning to classify chest X-rays as **Normal** or **Pneumonia**.
Simply upload or drag a chest X-ray image below to get started.
""")

# Sidebar for information
with st.sidebar:
    st.header("About")
    st.markdown("""
    - **Model**: Deep Learning CNN
    - **Input**: Chest X-ray images
    - **Output**: Classification (Normal/Pneumonia) with confidence
    - **Status**: ✅ Model loaded and ready
    """)

# File uploader
st.markdown("---")
st.subheader("Upload X-ray Image")
uploaded_file = st.file_uploader(
    "Drag and drop or click to upload",
    type=["jpg", "png", "jpeg"],
    help="Upload a chest X-ray image in JPG or PNG format"
)

if uploaded_file is not None:
    # Create two columns for image and results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("Analysis Results")
        
        if model is None:
            st.error("❌ Model file not found. Please ensure the model is at the correct path.")
        else:
            with st.spinner("Analyzing image..."):
                try:
                    prediction, confidence = predict(image, model)
                    
                    # Display results
                    if prediction == 1:  # Pneumonia
                        st.error(f"🚨 **Pneumonia Detected**")
                        st.metric("Confidence", f"{confidence:.1%}")
                    else:  # Normal
                        st.success(f"✅ **Normal**")
                        st.metric("Confidence", f"{confidence:.1%}")
                
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
    
    # Divider
    st.markdown("---")
    
    # File information
    st.caption(f" File: {uploaded_file.name} | Size: {uploaded_file.size / 1024:.2f} KB")
else:
    st.info(" Upload a chest X-ray image to begin analysis")