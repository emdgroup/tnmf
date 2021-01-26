"""
Author: Mathias Winkel
"""

from berkeley_images import load_images, plot_signal_reconstruction, plot_dictionary, \
    plot_activations, plot_partial_reconstruction, COLOR_SELECTIONS, close_figs, \
    plot_cost_function

from copy import deepcopy
from collections import defaultdict
import logging

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from TransformInvariantNMF import SparseNMF, ShiftInvariantNMF


COLOR_SELECTIONS_KEYS = list(COLOR_SELECTIONS.keys())


def st_define_dataset_params() -> dict:

    st.sidebar.markdown('# Image Dataset')

    return dict(
        path=st.sidebar.text_input('File path', value=r'BSR_bsds500/BSR/BSDS500/data/images/train'),
        pattern=st.sidebar.text_input('File filter', value='*.jpg'),
        max_images=st.sidebar.number_input('Max images', min_value=0, value=5),
        remove_margin=st.sidebar.number_input('Remove margin', min_value=0, value=0),
        color_mode=st.sidebar.radio('Channel(s)', COLOR_SELECTIONS_KEYS, 0),
        dtype=np.float64,
        filter=st.sidebar.checkbox('Low-pass/whitening filter', False),
    )

def st_define_nmf_params(image_shape: tuple) -> dict:

    st.sidebar.markdown('# NMF settings')

    # -------------------- general settings -------------------- #

    nmf_params = dict(
        verbose=st.sidebar.slider('Verbose', min_value=0, max_value=3, value=2),
        method=st.sidebar.radio('Mode', ['cachingFFT', 'fftconvolve', 'contract'], 0),
        reconstruction_mode=st.sidebar.radio('Reconstruction mode', ['full', 'valid', 'same'], 0),
        shift_invariant=st.sidebar.checkbox('Shift invariant', True),
        sparsity_H=st.sidebar.number_input('Activation sparsity', min_value=0.0, value=0.1),
        n_iterations=st.sidebar.number_input('# Iterations', min_value=1, value=5),
        refit_H=st.sidebar.checkbox('Refit activations without sparsity', True)
    )

    # -------------------- dictionary size  -------------------- #

    n_components = st.sidebar.number_input('# Dictionary elements', min_value=1, value=10)

    nmf_params.update(dict(
        n_components=n_components,
        atom_size=st.sidebar.number_input('Atom size', min_value=0, max_value=min(*image_shape), value=9),
    ))

    # -------------------- settings for shift invariance -------------------- #

    if nmf_params['shift_invariant']:

        st.sidebar.markdown('### Shift invariance settings')
        inhibition = st.sidebar.radio('Inhibition range', ['Auto', 'Manual'], 0)
        if inhibition == 'Auto':
            inhibition_range = None
        else:
            inhibition_range = st.sidebar.number_input('Inhibition range', min_value=0, value=atom_size)

        nmf_params.update(dict(
            inhibition_range=inhibition_range,
            inhibition_strength=st.sidebar.number_input('Inhibition strength', min_value=0.0, value=0.1)
        ))

    return nmf_params


@st.cache
def compute_nmf(V, nmf_params):
    """Streamlit caching of NMF fitting."""

    cost_function = defaultdict(list)

    def progress_callback(nmf: 'TransformInvariantNMF', i: int) -> bool:
        cost = nmf.cost_function()
        cost_str = str(cost).replace(', ', '\t')
        logging.info(f"Iteration: {i}\tCost function: {cost_str}")

        for key, value in cost.items():
            cost_function[key].append(value)

        cost_function['i'].append(i)

        return True

    nmf_params = nmf_params.copy()
    if nmf_params.pop('shift_invariant'):
        nmf = ShiftInvariantNMF(**nmf_params)
    else:
        nmf = SparseNMF(**nmf_params)
    nmf.fit(V, progress_callback)

    return nmf, cost_function


def st_plot(title, figs):
    st.markdown(title)
    for fig in figs:
        st.pyplot(fig)
    close_figs(figs)


if __name__ == '__main__':
    # -------------------- settings -------------------- #

    st.sidebar.markdown('# General settings')

    auto_update = st.sidebar.checkbox('Auto-Update', False)
    force_refresh = st.sidebar.button('Refresh')
    seed = st.sidebar.number_input('Random seed', value=42)
    np.random.seed(seed)

    dataset_params = st_define_dataset_params()

    logging.info(f'dataset params: {dataset_params}')

    images, image_shape = load_images(**dataset_params)

    nmf_params = st_define_nmf_params(image_shape)

    logging.info(f'NMF params: {nmf_params}')

    if not (auto_update or force_refresh):
        st.info('Auto-Update disabled')
        st.stop()

    # -------------------- model fitting -------------------- #

    # fit the NMF model
    nmf, cost_function = deepcopy(compute_nmf(images, nmf_params))

    # -------------------- visualization -------------------- #

    st_plot('Cost function', plot_cost_function(cost_function))

    color_mode = dataset_params["color_mode"]

    st_plot(f'# Learned dictionary: {color_mode}', plot_dictionary(nmf.W))

    st.markdown('# Signal reconstruction')

    # select signal to be visualized
    samples_per_image = 3 if color_mode == 'colors (identical basis)' else 1

    max_value = nmf.n_signals // samples_per_image - 1

    signal_number = 0 if nmf.n_signals == 1 else st.slider('Image number', min_value=0, max_value=max_value, value=0)

    st_plot('## Comparison to original signal',
            plot_signal_reconstruction(nmf, signal_number, samples_per_image))

    st_plot('## Activations',
            plot_activations(nmf.H, nmf_params, signal_number, samples_per_image))

    st_plot('## Partial Reconstructions',
            plot_partial_reconstruction(nmf, nmf_params, signal_number, samples_per_image))
