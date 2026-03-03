import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load trained model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

st.title("🧠 MNIST Handwritten Digit Recognition")

st.write("Draw a digit (0–9) below and click Predict")

# Canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data
        
        # Convert to grayscale
        img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY)
        
        # Resize to 28x28
        img = cv2.resize(img, (28, 28))
        
        # Normalize
        img = img / 255.0
        
        # Reshape for model
        img = img.reshape(1, 28, 28, 1)

        # Prediction
        prediction = model.predict(img)
        predicted_digit = np.argmax(prediction)

        st.success(f"Predicted Digit: {predicted_digit}")