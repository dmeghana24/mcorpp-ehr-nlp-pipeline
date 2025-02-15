
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("EHR NLP Pipeline Results Explorer")
df = pd.read_csv('results/model_comparison.csv')
st.write("## Sample predictions")
st.dataframe(df.sample(10))

label_counts = df['true'].value_counts()
st.write("### Label Distribution")
st.bar_chart(label_counts)

st.write("## Confusion Matrix")
st.image('figures/confusion_matrix.png')
st.write("## PR Curve")
st.image('figures/pr_curve.png')

idx = st.number_input("Row index", min_value=0, max_value=len(df)-1, value=0)
st.write(df.iloc[idx])
