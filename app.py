import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image

st.set_page_config(page_title="Plant AI", layout="centered")

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

h1 {
    text-align: center;
    font-size: 48px;
    background: -webkit-linear-gradient(#22c55e, #4ade80);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.upload-box {
    border: 2px dashed #22c55e;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    transition: 0.3s;
}
.upload-box:hover {
    background-color: rgba(34,197,94,0.1);
}

.result-card {
    padding: 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    margin-top: 20px;
    text-align: center;
    animation: fadeIn 1s ease-in-out;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}

.conf-bar {
    height: 10px;
    border-radius: 10px;
    background: linear-gradient(90deg, #22c55e, #4ade80);
    animation: grow 1.5s ease-in-out;
}

@keyframes grow {
    from {width: 0%;}
    to {width: 100%;}
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🌿 Plant Disease Detection</h1>", unsafe_allow_html=True)
st.write("### Upload a leaf image and let AI analyze it")

# Load model (fixed)
@st.cache_resource
def load_my_model():
    return load_model("plant_model.keras")

model = load_my_model()

# Hardcoded class names (IMPORTANT FIX)
class_names = [
    "Pepper bell Bacterial spot",
    "Pepper bell healthy",
    "Potato Early blight",
    "Potato Late blight",
    "Potato healthy",
    "Tomato Bacterial spot",
    "Tomato Early blight",
    "Tomato Late blight",
    "Tomato Leaf Mold",
    "Tomato Septoria leaf spot",
    "Tomato Spider mites",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato mosaic virus",
    "Tomato healthy"
]

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width='stretch')

    with st.spinner("🔍 AI is analyzing your leaf..."):
        img = image.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

    result = class_names[class_index]

    st.markdown(f"""
    <div class="result-card">
        <h2>🌱 {result}</h2>
        <p>Confidence: {round(confidence * 100, 2)}%</p>
        <div class="conf-bar" style="width:{int(confidence*100)}%"></div>
    </div>
    """, unsafe_allow_html=True)