import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(
    page_title="Fire Detection System",
    page_icon="🔥",
    layout="centered"
)

# Load Models (Cached)
@st.cache_resource
def load_models():
    cnn_model = load_model("models/fire_nofire_model.h5")
    rf_model = joblib.load("models/fire_nofire_rf_model.pkl")
    return cnn_model, rf_model

cnn_model, rf_model = load_models()

# Title & Description
st.title("🔥 Fire Detection using Deep Learning")
st.markdown(
    """
    This application detects **Fire / No Fire** from images using:
    - 🧠 **CNN (Deep Learning)**
    - 🌲 **Random Forest (ML baseline)**

    Upload an image to get predictions.
    """
)

# Image Upload
uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# Prediction Logic
def preprocess_image(img):
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Run Prediction
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img_array = preprocess_image(image)

    # CNN Prediction
    cnn_pred = cnn_model.predict(img_array)[0][0]
    cnn_result = "🔥 Fire" if cnn_pred < 0.5 else "✅ No Fire"

    # Random Forest Prediction
    rf_input = img_array.reshape(1, -1)
    rf_pred = rf_model.predict(rf_input)[0]
    rf_result = "🔥 Fire" if rf_pred == 0 else "✅ No Fire"

    # Results
    st.subheader("🔍 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="CNN Prediction",
            value=cnn_result
        )

    with col2:
        st.metric(
            label="Random Forest Prediction",
            value=rf_result
        )

    # Confidence Visualization
    st.subheader("📊 CNN Confidence")

    fig, ax = plt.subplots()
    ax.bar(
        ["Fire", "No Fire"],
        [1 - cnn_pred, cnn_pred]
    )
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1)

    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown(
    "**Developed by Keval Patel**  \n"
    "Machine Learning | Deep Learning | Computer Vision"
)
