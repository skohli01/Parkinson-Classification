import streamlit as st
from PIL import Image
import requests
from transformers import pipeline

# Load the Parkinson's classification pipeline
@st.cache_resource
def load_model():
  # pipe = pipeline("image-classification", "gianlab/swin-tiny-patch4-window7-224-finetuned-parkinson-classification")
    pipe = pipeline("image-classification", "skohli01/finetuned-parkinson-classification")
    return pipe
pipe = load_model()

st.title("Parkinson's Disease Classification")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Make prediction using the model
    if st.button("Predict"):
        with st.spinner("Classifying..."):
            prediction = pipe(image)
            # Get predicted label and score
            predicted_label = prediction[0]['label']
            score = prediction[0]['score']

            # Display prediction results
            st.write(f"**Prediction:** {predicted_label}")
            st.write(f"**Confidence:** {score:.2f}")
