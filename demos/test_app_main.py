"""A Streamlit app that displays the contents of a file"""

import os
import streamlit as st

obj = os.getenv("pablo_classifier_model_results")

st.title("Display workflow output")
st.write(obj.acc)
st.write(obj.model)
st.write(type(obj.model))
