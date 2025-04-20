import streamlit as st

st.title("About the Project")

st.header("Project Overview")
st.markdown("""
This project focuses on the automated classification of structural defects using deep learning.
We developed and evaluated three Convolutional Neural Networks (CNNs) to detect **cracks** in images of **walls**, **decks**, and **pavements**. The ultimate goal is to support early defect detection and reduce manual inspection time.
""")

st.header("Real-World Motivation")
st.markdown("""
Manual inspection of buildings and infrastructure is costly, time-consuming, and sometimes unreliable.
Automating defect detection can significantly enhance **safety**, improve **maintenance planning**, and reduce **operational costs** in the construction and civil engineering sectors.
""")

st.header("Aims and Hypothesis")
st.markdown("""
**Aims:**
- Train and evaluate three CNN models with differring complexity.
- Compare performance across three structural surface types.
- Build a user-facing tool using **Streamlit** for real-time predictions.

**Hypothesis:**
> Model 3, which balances complexity and regularization, will outperform simpler models in terms of accuracy and generalization.
""")

st.header("Tools and Technologies Used")
st.markdown("""
- **Languages & Frameworks:** Python, TensorFlow, Keras
- **Visualization:** Matplotlib, Seaborn, Plotly
- **App Deployment:** Streamlit
- **Other Tools:** scikit-learn, joblib, GitHub
""")

st.header("Dataset Summary")
st.markdown("""
The dataset consists of images categorized into:
- **Walls**, **Decks**, and **Pavements**
- Each image is labeled as **Cracked** or **Non-cracked**

Data preprocessing included resizing, label encoding, and class-specific augmentation for imbalance correction.
View the full analysis on the **Data Overview** page.
""")
st.header("Key Results")
st.markdown("""
- **Model 3** achieved the best overall performance:
  - **Test Accuracy:** ~85.8%
  - **Validation Accuracy:** ~85.8%
- Generalized well to Decks and Pavements despite training on Walls only.
- Deployed successfully in a functional Streamlit app.

*Try it out on the main page to test it yourself!*
""")

st.header("Acknowledgements")
st.markdown("""
- Image datasets sourced from Kaggle
- Inspired by similar ML image classification projects in civil engineering
""")