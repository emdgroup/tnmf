from copy import deepcopy
from itertools import cycle
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from tnmf.TransformInvariantNMF import TransformInvariantNMF
from tnmf.utils.data_loading import racoon_image
from tnmf.utils.demo import st_define_nmf_params, fit_nmf_model


def st_define_sample_params(verbose: bool = True) -> Tuple[dict, float]:
    # define image scale
    help_scale = \
        '''Downscaling the image allows for quicker interactive experimentation. Note that this does
        not change atom size.'''
    scale = st.sidebar.number_input('Image scale', min_value=0.05, value=0.25, step=0.05, help=help_scale)
    if verbose:
        st.sidebar.caption(help_scale)

    # define the number of channels
    help_channels = \
        '''The way, how color information in the image is processed. A grayscale image only has a single channel.
        A color images can be treated as a single, three-channel sample (leading to having three-channel, i.e.
        colorized, dictionary elements) or as three individual samples (leading to color-universal monochrome
        dictionary elements).'''
    channel_choices = {
        'grayscale':
            dict(get_v=lambda img: (np.dot(img, [0.2989, 0.5870, 0.1140]))[np.newaxis, np.newaxis, :, :],
                 #                      ^----simple grayscale conversion
                 restore=lambda X: (X[:, 0], ['Greys'])),
        'color, multi-channel':
            dict(get_v=lambda img: np.moveaxis(img, -1, 0)[np.newaxis, :, :, :],
                 restore=lambda X: (np.moveaxis(X, 1, -1), [None])),
        'color, one sample per channel':
            dict(get_v=lambda img: np.moveaxis(img, -1, 0)[:, np.newaxis, :, :],
                 restore=lambda X: (np.moveaxis(X, 1, -1), ['Reds', 'Greens', 'Blues'])),
    }
    channel_mode = st.sidebar.radio('# Channel mode', list(channel_choices.keys()), index=0, help=help_channels)
    channel_mode = channel_choices[channel_mode]
    if verbose:
        st.sidebar.caption(help_channels)

    return channel_mode, scale


def st_visualize_results(V: np.ndarray,
                         nmf: TransformInvariantNMF,
                         restore: Callable[[np.ndarray], np.ndarray],
                         verbose: bool = True):
    n_atoms = nmf.n_atoms
    V, Vc = restore(V)
    R, Rc = restore(nmf.R)

    st.markdown('# Input and Reconstruction')
    if verbose:
        st.caption('''The visualization below shows a **comparison between the input signal and its
        reconstruction** obtained through the learned factorization model.''')
    cols = st.columns(2)
    for col, X, Xc, title in zip(cols, [V, R], [Vc, Rc], ['Input', 'Reconstruction']):
        with col:
            for x, xc in zip(X, cycle(Xc)):
                fig = plt.figure()
                plt.imshow(x, cmap=xc)
                plt.title(f'{title}')
                st.pyplot(fig)

    st.markdown('# Learned Dictionary')
    if verbose:
        st.caption('''The visualization below shows an overview of all **learned dictionary atoms**.''')
    ncols = 5
    nrows = int(np.ceil(n_atoms / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    for w, ax in zip(nmf.W, axes.flatten()):
        ax.imshow(np.moveaxis(w, 0, -1) / w.max(), cmap='Greys' if w.shape[0] == 1 else None)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown('# Atoms, Activation and Partial Signal Reconstruction')
    if verbose:
        st.caption('''The visualization below shows the **learned dictionary atoms (left), their activations (center),
                   and their partial contributions (right)** to the reconstruction of an individual signal.''')

    def plot(X, title):
        fig = plt.figure()
        plt.imshow(X / X.max(), cmap='Greys' if X.ndim == 2 or (X.ndim == 3 and X.shape[2] == 1) else None)
        fig.suptitle(title)
        st.pyplot(fig)
        plt.close(fig)

    for i_atom in range(n_atoms):
        col1, col2, col3 = st.columns((1, 3, 3))
        with col1:
            plot(np.moveaxis(nmf.W[i_atom], 0, -1), f'Atom {i_atom}')
        with col2:
            plot(np.moveaxis(nmf.H[:, i_atom], 0, -1), f'Atom {i_atom} - Activation')
        with col3:
            plot(np.squeeze(np.moveaxis(nmf.R_partial(i_atom), (0, 1), (-2, -1))), f'Atom {i_atom} - Partial Reconstruction')


def main(progress_bar, verbose: bool = True):
    """
    Runs the streamlit demo on the famous scipy racoon demo image.

    Parameters
    ----------
    progress_bar
        Streamlit progress bar that needs to be updated during model fitting.
    verbose : bool
        If True, show detailed information.
    """

    channel_mode, scale = st_define_sample_params(verbose)

    # load the image
    img = racoon_image(gray=False, scale=scale)

    # samples are indexed V[sample_index, channel_index, sample_dimension_1, sample_dimension_2, ...]
    V = channel_mode['get_v'](img)

    # define the NMF parameters and fit the model
    default_nmf_params = {'n_atoms': 25, 'atom_shape': (10, 10)}
    nmf_params, fit_params = st_define_nmf_params(default_nmf_params, have_ground_truth=False, verbose=verbose)
    nmf = deepcopy(fit_nmf_model(V, nmf_params, fit_params, progress_bar))

    # visualize the results
    st_visualize_results(V, nmf, channel_mode['restore'], verbose)
