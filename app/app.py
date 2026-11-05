
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Fruit & Vegetable Classifier",
    page_icon="🍎",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/fruit_vegetable_model.keras"
    )

model = load_model()

# Load class names
with open("model/class_names.json", "r") as file:
    class_names = json.load(file)

# Application title
st.title("🍎 Fruit & Vegetable Classification")
st.write("Upload an image to identify the fruit or vegetable.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Predict button
    if st.button("Predict Image"):

        # Preprocess image
        image = image.resize((128, 128))
        image_array = np.array(image)

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Model prediction
        predictions = model.predict(image_array)

        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]

        confidence = predictions[0][predicted_index] * 100

        # Display result
        st.success(f"Prediction: {predicted_class.title()}")

        st.write(
            f"Confidence: {confidence:.2f}%"
        )
