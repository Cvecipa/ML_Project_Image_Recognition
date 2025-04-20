import streamlit as st
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

st.title("Model Evaluation")

st.markdown("""
This section displays saved training history and test set evaluation metrics
for each of the models trained on the **Walls** dataset.
""")

output_path = "outputs/v1"

eval_model_1 = joblib.load(os.path.join(output_path, "evaluation_tf_model_1_walls.pkl"))
eval_model_2 = joblib.load(os.path.join(output_path, "evaluation_tf_model_2_walls.pkl"))
eval_model_3 = joblib.load(os.path.join(output_path, "evaluation_tf_model_3_walls.pkl"))

histories = {
    "Model 1": joblib.load(os.path.join(output_path, "history_tf_model_1_walls.pkl")),
    "Model 2": joblib.load(os.path.join(output_path, "history_tf_model_2_walls.pkl")),
    "Model 3": joblib.load(os.path.join(output_path, "history_tf_model_3_walls.pkl"))
}

st.subheader("Test Set Evaluation")

results = pd.DataFrame({
    "Model": ["Model 1", "Model 2", "Model 3"],
    "Train Loss": [histories["Model 1"]['loss'][-1],
                   histories["Model 2"]['loss'][-1],
                   histories["Model 3"]['loss'][-1]],
    "Train Accuracy": [histories["Model 1"]['accuracy'][-1],
                       histories["Model 2"]['accuracy'][-1],
                       histories["Model 3"]['accuracy'][-1]],
    "Val Loss": [histories["Model 1"]['val_loss'][-1],
                 histories["Model 2"]['val_loss'][-1],
                 histories["Model 3"]['val_loss'][-1]],
    "Val Accuracy": [histories["Model 1"]['val_accuracy'][-1],
                     histories["Model 2"]['val_accuracy'][-1],
                     histories["Model 3"]['val_accuracy'][-1]],
    "Test Loss": [eval_model_1[0], eval_model_2[0], eval_model_3[0]],
    "Test Accuracy": [eval_model_1[1], eval_model_2[1], eval_model_3[1]]
})

st.dataframe(results.style.format({
    "Train Loss": "{:.4f}",
    "Train Accuracy": "{:.4f}",
    "Val Loss": "{:.4f}",
    "Val Accuracy": "{:.4f}",
    "Test Loss": "{:.4f}",
    "Test Accuracy": "{:.4f}"
}))

st.subheader("Training History")

model_descriptions = {
    "Model 1": "Model 1 includes regularization techniques such as L2, BatchNormalization, and Dropout. It maintained fairly stable accuracy, but fluctuating validation loss may indicate slight overfitting or instability.",
    "Model 2": "Model 2 is a simpler CNN with no regularization. It trained quickly and had a smaller architecture. Despite its simplicity, its performance was close to Model 1, which suggests it may generalize well with low computational cost.",
    "Model 3": "Model 3 strikes a balance between complexity and regularization. It showed the best overall accuracy and loss curves, suggesting this architecture is the most robust for the given task."
}

for model_name, history in histories.items():
    st.markdown(f"**{model_name}**")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.lineplot(ax=axes[0], data=history['loss'], label='Train')
    sns.lineplot(ax=axes[0], data=history['val_loss'], label='Validation')
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylim(0, 2)
    axes[0].legend()

    sns.lineplot(ax=axes[1], data=history['accuracy'], label='Train')
    sns.lineplot(ax=axes[1], data=history['val_accuracy'], label='Validation')
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    st.pyplot(fig)
    st.markdown(f"**Model Description:** {model_descriptions[model_name]}")
    st.markdown("---")