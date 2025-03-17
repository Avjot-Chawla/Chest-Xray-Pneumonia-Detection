import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# Load the model and set custom loss function if needed
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/xray_model.hdf5", compile=False)  # Load without compilation
    
    # Recompile the model with a valid loss function
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),  # Adjust optimizer as needed
        loss=tf.keras.losses.BinaryCrossentropy(reduction="sum_over_batch_size"),  # Correct loss function
        metrics=["accuracy"]
    )
    
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
    
    # Preprocess the image
    if uploaded_file is not None:
        # Display the uploaded image with use_container_width=True
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
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
