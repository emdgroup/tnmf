"""
Author: Mathias Winkel
"""

import os
import glob
import imageio
from copy import deepcopy
import logging
from more_itertools import chunked

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from TransformInvariantNMF import SparseNMF, ShiftInvariantNMF


def st_define_dataset_params() -> (str, str, int, int):
    st.sidebar.markdown('# Image Dataset')
    d = st.sidebar.text_input('File path', value=r'BSR_bsds500/BSR/BSDS500/data/images/train')
    f = st.sidebar.text_input('File filter', value='*.jpg')
    m = st.sidebar.number_input('Max images', min_value=0, value=5)
    c = st.sidebar.radio('Channel(s)', ['grey', 'red', 'green', 'blue', 'color'], 0)

    return d, f, m, c


def st_define_nmf_params(image_shape: tuple) -> dict:
    st.sidebar.markdown('# NMF settings')

    # -------------------- general settings -------------------- #

    nmf_params = dict(
        shift_invariant=st.sidebar.checkbox('Shift invariant', True),
        sparsity_H=st.sidebar.number_input('Activation sparsity', min_value=0.0, value=0.1),
        n_iterations=st.sidebar.number_input('# Iterations', min_value=1, value=5),
        refit_H=st.sidebar.checkbox('Refit activations without sparsity', True)
    )

    # -------------------- dictionary size  -------------------- #

    n_components = st.sidebar.number_input('# Dictionary elements', min_value=1, value=10)

    nmf_params.update(dict(
        n_components=n_components,
    ))

    # -------------------- settings for shift invariance -------------------- #

    if nmf_params['shift_invariant']:

        st.sidebar.markdown('### Shift invariance settings')
        atom_size = st.sidebar.number_input('Atom size', min_value=0, max_value=min(*image_shape), value=5)
        inhibition = st.sidebar.radio('Inhibition range', ['Auto', 'Manual'], 0)
        if inhibition == 'Auto':
            inhibition_range = None
        else:
            inhibition_range = st.sidebar.number_input('Inhibition range', min_value=0, value=atom_size)

        nmf_params.update(dict(
            atom_size=atom_size,
            inhibition_range=inhibition_range,
            inhibition_strength=st.sidebar.number_input('Inhibition strength', min_value=0.0, value=0.1)
        ))

    return nmf_params


@st.cache
def load_images(path: str, pattern: str, max_images: int = 0, color_mode: int = 'grey') -> (np.ndarray, tuple):
    images = []
    rows, cols = 10000, 10000

    for filename in glob.glob(os.path.join(os.getcwd(), path, pattern)):
        logging.info(filename)

        img = imageio.imread(filename)[:, :, :3] / 255.

        if color_mode == 'grey':
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])[:, :, np.newaxis]
        elif color_mode == 'red':
            img = img[..., 0:1]
        elif color_mode == 'green':
            img = img[..., 1:2]
        elif color_mode == 'blue':
            img = img[..., 2:3]
        elif color_mode == 'color':
            pass
        else:
            raise ValueError(f'Unsupported color mode: {color_mode}.')

        # flip all images to landscape orientation
        if img.shape[1] < img.shape[0]:
            img = np.swapaxes(img, 0, 1)

        images.append(img)

        rows, cols = min(rows, img.shape[0]), min(cols, img.shape[1])

        if 0 < max_images <= len(images):
            break

    # cut all images to fit the smallest image
    map(lambda img: img[:rows, :cols, :], images)

    images = np.array(images)

    # roll image index to the last dimension
    images = np.moveaxis(images, 0, -1)

    return images, (rows, cols)


@st.cache
def compute_nmf(V, nmf_params):
    """Streamlit caching of NMF fitting."""
    nmf_params = nmf_params.copy()
    if nmf_params.pop('shift_invariant'):
        nmf = ShiftInvariantNMF(**nmf_params)
    else:
        nmf = SparseNMF(**nmf_params)
    nmf.fit(V)
    logging.info('NMF finished')
    return nmf


def st_compare_single_signals(nmf, signal_number):
    st.markdown('# Signal reconstruction')

    col1, col2 = st.beta_columns(2)
    with col1:
        fig = plt.figure()
        plt.imshow(nmf.V[..., signal_number])
        plt.grid(False)
        st.pyplot(fig)
    with col2:
        fig = plt.figure()
        plt.imshow(np.clip(nmf.R[..., signal_number], 0., 1.))
        plt.grid(False)
        st.pyplot(fig)


def st_show_dictionary(W, color_mode):
    def plot_dictionary(W, channel):
        figs = []
        for m in range(W.shape[-1]):
            fig = plt.figure()
            plt.imshow(W[..., channel, m])
            plt.grid(False)
            plt.axis('off')
            plt.tight_layout()
            figs.append(fig)
        return figs

    st.markdown('# Learned dictionary')
    for channel in range(W.shape[-2]):
        if color_mode == 'color':
            color = ['red', 'green', 'blue'][channel]
        else:
            color = color_mode

        st.markdown(f'## {color}')
        figs = plot_dictionary(W, channel)
        for row in chunked(figs, 10):
            cols = st.beta_columns(10)
            for fig, col in zip(row, cols):
                with col:
                    st.pyplot(fig)


def st_plot_activations(H, nmf_params, signal_number):

    st.markdown('# Activation pattern')

    m = st.slider('Dictionary atom', min_value=0, max_value=H.shape[-2]-1)
    fig = plt.figure(figsize=(10, 5))
    if nmf_params['shift_invariant']:
        plt.imshow(H[..., m, signal_number], aspect='auto')
    else:
        plt.imshow(H[0], aspect='auto')
    plt.colorbar()
    plt.grid(False)
    st.pyplot(fig)


if __name__ == '__main__':
    # -------------------- settings -------------------- #

    st.sidebar.markdown('# General settings')

    seed = st.sidebar.number_input('Random seed', value=42)
    np.random.seed(seed)

    d, f, max_images, color_mode = st_define_dataset_params()
    images, image_shape = load_images(d, f, max_images, color_mode)

    nmf_params = st_define_nmf_params(image_shape)

    # -------------------- model fitting -------------------- #

    # fit the NMF model
    nmf = deepcopy(compute_nmf(images, nmf_params))

    # -------------------- visualization -------------------- #

    # select signal to be visualized
    signal_number = 0 if nmf.n_signals == 1 else st.slider('Image number', min_value=0, max_value=nmf.n_signals - 1,
                                                           value=0)

    # show reconstruction of individual signals
    st_compare_single_signals(nmf, signal_number)

    # show learned dictionary
    st_show_dictionary(nmf.W, color_mode)

    # show activation pattern
    st_plot_activations(nmf.H, nmf_params, signal_number)
