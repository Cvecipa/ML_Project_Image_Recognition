import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import joblib
import os
import pandas as pd
import plotly.express as px

MODEL_PATHS = {
    "Model 1": "outputs/v1/tf_model_1_walls.h5",
    "Model 2": "outputs/v1/tf_model_2_walls.h5",
    "Model 3": "outputs/v1/tf_model_3_walls.h5"
}

CLASS_INDEX_PATH = "outputs/v1/class_indices.pkl"
IMAGE_SHAPE_PATH = "outputs/v1/image_shape.pkl"

class_indices = joblib.load(CLASS_INDEX_PATH)
image_shape = joblib.load(IMAGE_SHAPE_PATH)
target_map = {v: k for k, v in class_indices.items()}

st.title("Structural Defect Classifier")
st.markdown("Upload an image of a structure to predict if it's **Cracked** or **Non-cracked**.")

model_choice = st.sidebar.selectbox("Select a Model", list(MODEL_PATHS.keys()))
model = tf.keras.models.load_model(MODEL_PATHS[model_choice])

uploaded_file = st.file_uploader("Upload a structural image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    img = image.load_img(uploaded_file, target_size=image_shape[:2])
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    proba = model.predict(img_array)[0][0]
    pred_class = target_map[int(proba < 0.5)]  

    st.subheader("Prediction Result")
    st.write(f"**Predicted Class:** {pred_class}")
    st.write(f"**Confidence:** {1 - proba if pred_class == 'Cracked' else proba:.2f}")

    prob_df = pd.DataFrame([[1 - proba, proba]], columns=['Cracked', 'Non-cracked'])
    prob_df = prob_df.round(3)
    prob_df = pd.melt(prob_df)
    fig = px.bar(prob_df, x="variable", y="value", labels={"value": "Probability", "variable": "Class"},
                 title="Prediction Probability", range_y=[0, 1])
    st.plotly_chart(fig)