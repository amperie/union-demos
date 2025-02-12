"""A Streamlit app that displays the contents of a file"""

import os
import streamlit as st

obj = os.getenv("CLS_MODEL_RESULTS")

st.title("Display workflow output")
st.write(obj.acc)
st.write(obj.model)
st.write(type(obj.model))
