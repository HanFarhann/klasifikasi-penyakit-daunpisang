import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ========== KONFIGURASI ==========
MODEL_PATH = "models/best_model.keras"   # Sesuaikan dengan path model kamu

CLASS_NAMES = [
    "bukan_daun_pisang",
    "cordana",
    "healthy",
    "pestalotiopsis",
    "sigatoka"
]

DISEASE_INFO = {
    "bukan_daun_pisang": "Gambar ini bukan daun pisang.",
    "cordana": "Penyakit Cordana dapat menyebabkan bercak cokelat pada daun pisang.",
    "healthy": "Daun pisang terlihat sehat, tidak ada gejala penyakit.",
    "pestalotiopsis": "Pestalotiopsis dapat menyebabkan bercak tidak beraturan pada daun.",
    "sigatoka": "Sigatoka adalah penyakit serius pada pisang yang menimbulkan bercak hitam."
}

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ========== FUNGSI PREPROCESS ==========
def preprocess_image(image):
    img = image.resize((224, 224))  # Sesuaikan dengan ukuran input model
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ========== STREAMLIT UI ==========
st.title("🍌 Deteksi Penyakit Daun Pisang")
st.write("Upload gambar daun pisang untuk mendeteksi penyakit.")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    if st.button("Prediksi"):
        with st.spinner("Sedang memproses..."):
            img_array = preprocess_image(image)
            predictions = model.predict(img_array)
            predicted_class = CLASS_NAMES[np.argmax(predictions)]
            confidence = np.max(predictions)

        st.subheader(f"Hasil Prediksi: **{predicted_class}**")
        st.write(f"Tingkat keyakinan: **{confidence:.2f}**")
        st.info(DISEASE_INFO[predicted_class])
