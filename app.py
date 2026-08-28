import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

MODEL_PATH = "model/freshness_model.keras"
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="Fruit & Vegetable Freshness Classifier",
    page_icon="🍎",
    layout="centered"
)

st.title("🍎 Fruit & Vegetable Freshness Classifier")
st.write("Upload an image to predict whether the fruit or vegetable is **Fresh** or **Rotten**.")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if model is None:
        st.warning(
            "The trained model is not available yet. "
            "Run `python train_model.py` after adding your dataset."
        )
    else:
        img = image.resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)

        prediction = float(model.predict(arr, verbose=0)[0][0])

        # 0 = Fresh, 1 = Rotten
        if prediction >= 0.5:
            label = "Rotten"
            confidence = prediction * 100
            icon = "🔴"
        else:
            label = "Fresh"
            confidence = (1 - prediction) * 100
            icon = "🟢"

        st.subheader(f"{icon} Prediction: {label}")
        st.metric("Confidence", f"{confidence:.2f}%")
