import numpy as np
from importlib import import_module

import streamlit as st
from pathlib import Path

# show TNMF header
img_file = Path(__file__).parent.parent / 'doc' / 'logos' / 'tnmf_header.png'
st.image(str(img_file), use_column_width='always')

# mapping of streamlit demo names to the corresponding demo file and optional parameters
DEMO_NAME_DICT = {
    '1-D Synthetic Signals': ('synthetic_signals', {'n_dims': 1}),
    '2-D Synthetic Signals': ('synthetic_signals', {'n_dims': 2}),
}

# create progress bar on the top that is shown for all demos
progress_bar = st.sidebar.progress(1.)

# toggle verbose mode
help_verbose = 'Displays / hides **detailed explanations**.'
verbose = st.sidebar.checkbox('Verbose', True, help=help_verbose)
if verbose:
    st.sidebar.caption(help_verbose)

# select the demo
help_select_demo = 'The specific **demo example** that gets executed.'
selected_demo = st.sidebar.selectbox('Demo example', list(DEMO_NAME_DICT.keys()), 1, help=help_select_demo)
if verbose:
    st.sidebar.caption(help_select_demo)

# select the random seed
help_seed = 'The fixed **random seed** that is used for the simulation.'
seed = st.sidebar.number_input('Random seed', value=42, help=help_seed)
np.random.seed(seed)
if verbose:
    st.sidebar.caption(help_seed)

# extract the demo name and parameters, import the corresponding module, and execute the demo
demo_name, demo_args = DEMO_NAME_DICT[selected_demo]
demo_module = import_module(demo_name)
demo_module.main(progress_bar, verbose=verbose, **demo_args)
