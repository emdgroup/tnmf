import numpy as np
from importlib import import_module

import streamlit as st

# mapping of streamlit demo names to the corresponding demo file and optional parameters
DEMO_NAME_DICT = {
    '1-D Synthetic Signals': ('synthetic_signals', {'n_dims': 1}),
}

# create progress bar on the top that is shown for all demos
progress_bar = st.sidebar.progress(1.)

# select the demo and random seed
selected_demo = st.sidebar.selectbox('Select demo', list(DEMO_NAME_DICT.keys()), 0)
seed = st.sidebar.number_input('Random seed', value=42)
np.random.seed(seed)

# extract the demo name and parameters, import the corresponding module, and execute the demo
demo_name, demo_args = DEMO_NAME_DICT[selected_demo]
demo_module = import_module(demo_name)
demo_module.main(progress_bar, **demo_args)
