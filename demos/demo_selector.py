import sys

import importlib.resources
from importlib import import_module

import numpy as np
import streamlit as st

# mapping of streamlit demo names to the corresponding demo file and optional parameters
DEMO_NAME_DICT = {
    '1-D Synthetic Signals': ('synthetic_signals', {'n_dims': 1}),
    '2-D Synthetic Signals': ('synthetic_signals', {'n_dims': 2}),
    'Racoon Image': ('demo_image', {}),
}


def main(demo_name: str):
    # show TNMF header
    with importlib.resources.path('logos', 'tnmf_header.png') as img_file:
        st.image(str(img_file), use_column_width='always')

    # create progress bar on the top that is shown for all demos
    progress_bar = st.sidebar.progress(1.)

    # toggle verbose mode
    help_verbose = 'Displays / hides **detailed explanations**.'
    verbose = st.sidebar.checkbox('Verbose', True, help=help_verbose)
    if verbose:
        st.sidebar.caption(help_verbose)

    # select the demo
    help_select_demo = 'The specific **demo example** that gets executed.'
    default_demo = list(DEMO_NAME_DICT.keys()).index(demo_name)
    selected_demo = st.sidebar.selectbox('Demo example', list(DEMO_NAME_DICT.keys()), default_demo, help=help_select_demo)
    if verbose:
        st.sidebar.caption(help_select_demo)

    # select the random seed
    help_seed = 'The fixed **random seed** that is used for the simulation.'
    seed = st.sidebar.number_input('Random seed', value=42, help=help_seed)
    np.random.seed(seed)
    if verbose:
        st.sidebar.caption(help_seed)

    # show demo info text
    if verbose:
        st.markdown('''
        ## Demo guide
        This dashboard demonstrates the use of the *Transform-Invariant Non-Negative Matrix Factorization (TNMF) package*
        in the specific context of learning **shift-invariant representations**.
        Via the **drop-down menu** on the left, you can choose between different demo examples.

        ## Usage
        Detailed explanations of the available options and the shown output (including this text) can be displayed or hidden
        using the **'Verbose' checkbox**. When the checkbox is unticked, the description of the control widgets can still
        be accessed via the **tooltips** next to them.
        ''')

    # extract the demo name and parameters, import the corresponding module, and execute the demo
    demo_name, demo_args = DEMO_NAME_DICT[selected_demo]
    demo_module = import_module(demo_name)
    demo_module.main(progress_bar, verbose=verbose, **demo_args)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main('2-D Synthetic Signals')
