import streamlit as st
import numpy as np
import pandas as pd

# Title
st.title("My First Streamlit App")

# Text
st.write("Hello, Streamlit!")

# Slider
number = st.slider("Pick a number", 0, 100)
st.write(f"Selected number: {number}")

# DataFrame
data = pd.DataFrame(
    np.random.randn(10, 3),
    columns=['A', 'B', 'C']
)
st.line_chart(data)

# Run:  streamlit run streamlit.py --server.port 8502
def title(param):
    return None


def selectbox(param, param1):
    return None