import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image
from collections import Counter

st.title("Actual Image Count from Dataset Folders")

base_dir = "inputs/cracks_dataset_new"  

structure_types = ["Walls", "Decks", "Pavements"]
subsets = ["train", "val", "test"]
class_names = ["Cracked", "Non-cracked"]

data = []

for structure in structure_types:
    for subset in subsets:
        for class_name in class_names:
            folder = os.path.join(base_dir, subset, structure, class_name)
            if os.path.exists(folder):
                count = len([
                    f for f in os.listdir(folder)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
                data.append({
                    "Structure": structure,
                    "Subset": subset,
                    "Class": class_name,
                    "Image Count": count
                })

df = pd.DataFrame(data)

st.subheader("Image Count by Class, Structure, and Subset")
st.dataframe(df)

st.subheader("Interactive Class Distribution per Structure and Subset")

for structure in structure_types:
    st.markdown(f"**{structure} Class Distribution**")
    sub_df = df[df["Structure"] == structure]
    fig = px.bar(
        sub_df,
        x="Subset",
        y="Image Count",
        color="Class",
        barmode="group",
        title=f"{structure} - Cracked vs Non-cracked per Subset",
        labels={"Image Count": "Number of Images"},
        text="Image Count"
    )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Total Image Count per Structure")

df_total = df.groupby("Structure")["Image Count"].sum().reset_index()

fig_total = px.bar(
    df_total,
    x="Structure",
    y="Image Count",
    text="Image Count",
    title="Total Images per Structure",
    labels={"Image Count": "Total Images"}
)

fig_total.update_traces(textposition='outside')
fig_total.update_layout(yaxis_range=[0, df_total["Image Count"].max() + 200])

st.plotly_chart(fig_total)

st.subheader("Image Resolution and Format Insights")

resolution_data = []
format_data = []

for structure in structure_types:
    for subset in subsets:
        for class_name in class_names:
            folder = os.path.join(base_dir, subset, structure, class_name)
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        try:
                            img_path = os.path.join(folder, file)
                            with Image.open(img_path) as img:
                                resolution = img.size  # (width, height)
                                fmt = img.format
                                resolution_data.append(resolution)
                                format_data.append(fmt)
                        except Exception as e:
                            st.warning(f"Error reading {file}: {e}")

res_df = pd.DataFrame(resolution_data, columns=["Width", "Height"])
res_df["Resolution"] = res_df["Width"].astype(str) + " x " + res_df["Height"].astype(str)

common_resolutions = res_df["Resolution"].value_counts().reset_index()
common_resolutions.columns = ["Resolution", "Count"]

fig_res = px.bar(
    common_resolutions.head(10),
    x="Resolution",
    y="Count",
    title="Top 10 Most Common Image Resolutions",
    text="Count"
)
fig_res.update_traces(textposition="outside")
st.plotly_chart(fig_res)

format_counts = Counter(format_data)
format_df = pd.DataFrame(format_counts.items(), columns=["Format", "Count"])

fig_fmt = px.pie(
    format_df,
    names="Format",
    values="Count",
    title="Image Format Distribution"
)
st.plotly_chart(fig_fmt)