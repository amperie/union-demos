"""A Streamlit app that displays the contents of a file"""
import os

current_directory = os.getcwd()
print(current_directory)
from os import walk

filenames = next(walk("/root"), (None, None, []))[2]  # [] if no file
import sys
pythonpath = sys.path

list_subfolders_with_paths = [f.path for f in os.scandir("/root") if f.is_dir()]

import streamlit as st
from demos.tasks.dataclass_defs import HpoResults

obj = os.getenv("CLS_MODEL_RESULTS")
obj = HpoResults.from_flytedir(obj)

st.title("Display workflow output")
st.write(obj.acc)
st.write(obj.model)
st.write(obj.data)
st.write(pythonpath)
st.write(current_directory)
st.write(filenames)
st.write(list_subfolders_with_paths)
