import streamlit as st
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

st.title("Model Comparison")

st.markdown("""
This section compares the three trained CNN models across multiple criteria:
- Test set performance
- Training and validation metrics
- Final epoch performance
- Model complexity
""")

output_path = "outputs/v1"

eval_1 = joblib.load(os.path.join(output_path, "evaluation_tf_model_1_walls.pkl"))
eval_2 = joblib.load(os.path.join(output_path, "evaluation_tf_model_2_walls.pkl"))
eval_3 = joblib.load(os.path.join(output_path, "evaluation_tf_model_3_walls.pkl"))

hist_1 = joblib.load(os.path.join(output_path, "history_tf_model_1_walls.pkl"))
hist_2 = joblib.load(os.path.join(output_path, "history_tf_model_2_walls.pkl"))
hist_3 = joblib.load(os.path.join(output_path, "history_tf_model_3_walls.pkl"))

st.subheader("Metric Focus")

available_metrics = ["Test Accuracy", "Test Loss", "Train Accuracy", "Val Accuracy", "Train Loss", "Val Loss"]
metric_focus = st.selectbox("Select a metric to highlight:", available_metrics)

df_all = pd.DataFrame({
    "Model": ["Model 1", "Model 2", "Model 3"],
    "Test Loss": [eval_1[0], eval_2[0], eval_3[0]],
    "Test Accuracy": [eval_1[1], eval_2[1], eval_3[1]],
    "Train Accuracy": [hist_1['accuracy'][-1], hist_2['accuracy'][-1], hist_3['accuracy'][-1]],
    "Val Accuracy": [hist_1['val_accuracy'][-1], hist_2['val_accuracy'][-1], hist_3['val_accuracy'][-1]],
    "Train Loss": [hist_1['loss'][-1], hist_2['loss'][-1], hist_3['loss'][-1]],
    "Val Loss": [hist_1['val_loss'][-1], hist_2['val_loss'][-1], hist_3['val_loss'][-1]]
})

if "Loss" in metric_focus:
    best_model_idx = df_all[metric_focus].idxmin()
else:
    best_model_idx = df_all[metric_focus].idxmax()

best_model_name = df_all.loc[best_model_idx, "Model"]
best_metric_value = df_all.loc[best_model_idx, metric_focus]

st.metric(label=best_model_name, value=f"{best_metric_value:.4f}")
highlight_df = df_all[["Model", metric_focus]].rename(columns={metric_focus: f"{metric_focus} (Focused)"})
st.dataframe(highlight_df.style.format({f"{metric_focus} (Focused)": "{:.4f}"}))

st.subheader("1. Test Set Performance Summary")

comparison_df = df_all[["Model", "Test Loss", "Test Accuracy"]]
st.dataframe(comparison_df.style.format({"Test Loss": "{:.4f}", "Test Accuracy": "{:.4f}"}))

fig = px.bar(
    comparison_df,
    x="Model",
    y="Test Accuracy",
    title="Test Accuracy per Model",
    text="Test Accuracy",
    range_y=[0, 1]
)
fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
fig.update_layout(yaxis_title="Accuracy", xaxis_title="Model")
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Interpretation:**  
Model 3 outperforms both Model 1 and Model 2 with the lowest test loss and highest accuracy, showcasing the best generalization capabilities.
""")

st.subheader("2. Train and Validation Metrics")

metrics_df = df_all[["Model", "Train Accuracy", "Val Accuracy", "Train Loss", "Val Loss"]]
st.dataframe(metrics_df.style.format({
    "Train Accuracy": "{:.4f}",
    "Val Accuracy": "{:.4f}",
    "Train Loss": "{:.4f}",
    "Val Loss": "{:.4f}"
}))

acc_df = metrics_df[["Model", "Train Accuracy", "Val Accuracy"]].melt(id_vars="Model", var_name="Type", value_name="Accuracy")

fig2 = px.bar(
    acc_df,
    x="Model",
    y="Accuracy",
    color="Type",
    barmode="group",
    text="Accuracy",
    title="Train vs Validation Accuracy",
    range_y=[0, 1]
)
fig2.update_traces(texttemplate='%{text:.4f}', textposition='outside')
fig2.update_layout(yaxis_title="Accuracy", xaxis_title="Model")
st.plotly_chart(fig2)

st.markdown("""
**Interpretation:**  
This breakdown shows the learning behavior of each model. Model 3 maintains a strong balance and high accuracy across both training and validation, indicating effective generalization.
""")

st.subheader("3. Final Epoch Loss and Accuracy Comparison")

final_df = pd.DataFrame({
    "Model": ["Model 1", "Model 2", "Model 3"],
    "Final Train Loss": [hist_1['loss'][-1], hist_2['loss'][-1], hist_3['loss'][-1]],
    "Final Val Loss": [hist_1['val_loss'][-1], hist_2['val_loss'][-1], hist_3['val_loss'][-1]],
    "Final Train Accuracy": [hist_1['accuracy'][-1], hist_2['accuracy'][-1], hist_3['accuracy'][-1]],
    "Final Val Accuracy": [hist_1['val_accuracy'][-1], hist_2['val_accuracy'][-1], hist_3['val_accuracy'][-1]]
})
st.dataframe(final_df.style.format({
    "Final Train Loss": "{:.4f}",
    "Final Val Loss": "{:.4f}",
    "Final Train Accuracy": "{:.4f}",
    "Final Val Accuracy": "{:.4f}"
}))

fig3 = px.bar(
    final_df,
    x="Model",
    y="Final Val Accuracy",
    text="Final Val Accuracy",
    title="Final Validation Accuracy",
    range_y=[0, 1]
)
fig3.update_traces(texttemplate='%{text:.4f}', textposition='outside')
fig3.update_layout(yaxis_title="Accuracy", xaxis_title="Model")
st.plotly_chart(fig3)

st.markdown("""
**Interpretation:**  
Model 3 demonstrates the most stable and optimal end-of-training performance. It consistently leads in accuracy while keeping loss values low, making it the strongest candidate overall.
""")

st.subheader("4. Model Complexity (Parameter Count)")

params_df = pd.DataFrame({
    "Model": ["Model 1", "Model 2", "Model 3"],
    "Trainable Parameters": [295000, 123000, 165000]
})
st.dataframe(params_df)

fig4 = px.bar(
    params_df,
    x="Model",
    y="Trainable Parameters",
    text="Trainable Parameters",
    title="Model Complexity (Trainable Parameters)"
)
fig4.update_traces(texttemplate='%{text}', textposition='outside')
fig4.update_layout(yaxis_title="Parameters", xaxis_title="Model")
st.plotly_chart(fig4)

st.markdown("""
**Interpretation:**  
Model 2 is the simplest with the fewest parameters, but Model 3 offers the best performance-to-complexity ratio, balancing accuracy with moderate complexity.
""")

st.markdown("---")