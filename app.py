import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas

def preprocess_image_advanced(img):
    """Advanced preprocessing for real-world handwritten digit images"""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Check if image is inverted (dark digit on light background)
    mean_intensity = np.mean(img)
    if mean_intensity > 127:
        # Invert the image (MNIST expects white digits on black background)
        img = 255 - img
    
    # Apply thresholding to clean up the image
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Remove noise
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    # Find the bounding box of the digit
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Add padding
        padding = 4
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2 * padding)
        h = min(img.shape[0] - y, h + 2 * padding)
        
        # Crop to digit
        img = img[y:y+h, x:x+w]
    
    # Resize to 20x20 while maintaining aspect ratio
    h, w = img.shape
    if h > w:
        new_h = 20
        new_w = int(w * 20 / h)
    else:
        new_w = 20
        new_h = int(h * 20 / w)
    
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Center in 28x28 image
    img_centered = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    img_centered[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
    
    # Normalize
    img_normalized = img_centered.astype('float32') / 255.0
    
    return img_normalized, img_centered

# Load trained model
try:
    model = tf.keras.models.load_model("mnist_cnn_model.h5")
    model_type = "CNN"
except:
    model = tf.keras.models.load_model("mnist_model.h5")
    model_type = "Dense"

st.title("🧠 MNIST Handwritten Digit Recognition")
st.caption(f"Using {model_type} Model")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["✏️ Draw Digit", "📁 Upload Image"])

with tab1:
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

    if st.button("Predict", key="predict_canvas"):
        if canvas_result.image_data is not None:
            img = canvas_result.image_data
            
            # Convert to grayscale
            img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY)
            
            # Resize to 28x28
            img = cv2.resize(img, (28, 28))
            
            # Normalize
            img = img / 255.0
            
            # Prepare input based on model type
            if model_type == "CNN":
                img = img.reshape(1, 28, 28, 1)
            else:
                img = img.reshape(1, 28 * 28)

            # Prediction
            prediction = model.predict(img, verbose=0)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            st.success(f"Predicted Digit: **{predicted_digit}** (Confidence: {confidence:.2f}%)")

with tab2:
    st.write("Upload an image of a handwritten digit (0–9)")
    st.info("💡 **Tip:** Works best with clear images of single digits. Dark or light backgrounds both supported!")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "bmp"])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        # Process image with advanced preprocessing
        img_array = np.array(image)
        img_normalized, img_processed = preprocess_image_advanced(img_array)
        
        # Display images
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(image, caption="Original Image", width=150)
        
        with col2:
            st.image(img_processed, caption="Preprocessed", width=150, clamp=True)
        
        # Prepare input based on model type
        if model_type == "CNN":
            img_final = img_normalized.reshape(1, 28, 28, 1)
        else:
            img_final = img_normalized.reshape(1, 28 * 28)

        # Make prediction
        prediction = model.predict(img_final, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        with col3:
            st.metric(label="Prediction", value=predicted_digit)
            st.metric(label="Confidence", value=f"{confidence:.1f}%")
        
        # Show detailed predictions
        if confidence < 70:
            st.warning(f"⚠️ Low confidence ({confidence:.1f}%). Try a clearer image!")
        else:
            st.success(f"✓ Predicted Digit: **{predicted_digit}** with {confidence:.1f}% confidence")
        
        # Show prediction probabilities
        st.write("**Prediction Probabilities:**")
        prob_data = {f"{i}": prediction[0][i] for i in range(10)}
        st.bar_chart(prob_data)