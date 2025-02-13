import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# Load the model and set custom loss function if needed
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/xray_model.hdf5")  # Load the model as usual
    
    # If the model has an invalid 'reduction' argument, we fix it here
    # Check if the model has the loss function with the invalid reduction and replace it
    for layer in model.layers:
        if hasattr(layer, 'loss') and layer.loss == 'auto':
            layer.loss = 'mean'  # Replace with a valid value such as 'mean', 'sum', etc.
    
    return model

# Image preprocessing for the model input
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))  # Assuming the model expects 224x224 images
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict pneumonia from the X-ray image
def predict(model, img_array):
    prediction = model.predict(img_array)
    # Assuming the model outputs probabilities for binary classification (pneumonia vs. normal)
    return np.argmax(prediction, axis=-1)[0]  # Returns 0 or 1

# Web app structure
def main():
    st.title("Chest X-Ray Pneumonia Detection")
    
    # Upload X-ray image
    uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image with use_container_width=True
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)  # Fixed the warning
        
        # Preprocess the image
        img_array = preprocess_image(uploaded_file)
        
        # Load the model
        model = load_model()
        
        # Predict the result
        if st.button("Predict Pneumonia"):
            prediction = predict(model, img_array)
            if prediction == 1:
                st.write("The X-ray shows signs of Pneumonia.")
            else:
                st.write("The X-ray is Normal.")
    
    st.text("This app is built using Streamlit and TensorFlow.")

if __name__ == "__main__":
    main()
