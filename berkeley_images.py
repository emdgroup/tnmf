"""
Author: Mathias Winkel
"""

import os
import glob
import imageio
from copy import deepcopy
import logging

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from TransformInvariantNMF import SparseNMF, ShiftInvariantNMF

COLOR_SELECTIONS = {
    'grey': lambda img: [np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])[:, :, np.newaxis]],
    'red': lambda img: [img[..., 0:1]],
    'green': lambda img: [img[..., 1:2]],
    'blue': lambda img: [img[..., 2:3]],
    'color basis': lambda img: [img],
    'colors (identical basis)': lambda img: [img[..., 0:1], img[..., 1:2], img[..., 2:3]],
}

COLOR_SELECTIONS_KEYS = list(COLOR_SELECTIONS.keys())


def st_define_dataset_params() -> (str, str, int, int):
    st.sidebar.markdown('# Image Dataset')
    d = st.sidebar.text_input('File path', value=r'BSR_bsds500/BSR/BSDS500/data/images/train')
    f = st.sidebar.text_input('File filter', value='*.jpg')
    m = st.sidebar.number_input('Max images', min_value=0, value=5)
    c = st.sidebar.radio('Channel(s)', COLOR_SELECTIONS_KEYS, 0)

    return d, f, m, c


def st_define_nmf_params(image_shape: tuple) -> dict:

    st.sidebar.markdown('# NMF settings')

    # -------------------- general settings -------------------- #

    nmf_params = dict(
        verbose=st.sidebar.slider('Verbose', min_value=0, max_value=3, value=2),
        use_fft=st.sidebar.checkbox('Use FFT', True),
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


def load_images(path: str, pattern: str, max_images: int = 0,
                color_mode: int = 'grey', dtype=np.float32) -> (np.ndarray, tuple):
    images = []
    rows, cols = 10000, 10000

    logging.info(f'Loading files from "{path}" with pattern "{pattern}" in color mode "{color_mode}"')

    count = 0
    for count, filename in enumerate(glob.glob(os.path.join(os.getcwd(), path, pattern))):
        # limit number of images
        if 0 < max_images <= count:
            break

        logging.info(filename)

        img = (imageio.imread(filename)[:, :, :3] / 255.).astype(dtype)
        channels = COLOR_SELECTIONS[color_mode](img)

        # flip all images to landscape orientation
        channels = [np.swapaxes(ch, 0, 1) if ch.shape[1] < ch.shape[0] else ch for ch in channels]

        for channel in channels:
            rows, cols = min(rows, channel.shape[0]), min(cols, channel.shape[1])
            images.append(channel)

    # cut all images to fit the smallest image
    images = [image[:rows, :cols, :] for image in images]

    images = np.asarray(images, dtype=dtype)

    # roll image index to the last dimension because this is how the NMF needs its data
    images = np.moveaxis(images, 0, -1)

    # just ensure nothing was garbled in the code above
    assert (images.ndim == 4)
    assert (images.shape[0] == rows and images.shape[1] == cols)
    assert (images.shape[2] in (1, 3))
    assert (images.shape[3] in (count, 3 * count))
    assert (images.dtype == dtype)

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
    return nmf


def plot_signal_reconstruction(nmf, signal_number, samples_per_image):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
    for iplot, ax in enumerate(axes):
        if iplot == 0:
            data = np.squeeze(nmf.V[..., signal_number * samples_per_image:(signal_number + 1) * samples_per_image])
        else:
            data = np.clip(np.squeeze(nmf.R[..., signal_number * samples_per_image:(signal_number + 1) * samples_per_image]), 0., 1.)

        ax.imshow(data)
        ax.grid(False)

    return fig


def st_show_signal_reconstruction(figs):
    st.markdown('## Comparison to original signal')
    st.pyplot(figs)


def plot_dictionary(W, num_columns=10):
    nrows = (W.shape[-1] + num_columns - 1) // num_columns
    fig = plt.figure(figsize=(num_columns, nrows))

    for m in range(W.shape[-1]):
        ax = fig.add_subplot(nrows, num_columns, m+1, xticks=[], yticks=[])
        ax.imshow(W[..., m])
        ax.set_title(f'{m}')
        ax.grid(False)

    plt.tight_layout()
    return fig


def st_show_dictionary(fig):
    st.markdown(f'# Learned dictionary: {color_mode}')
    st.pyplot(fig, clear_figure=False)


def plot_activations(H, nmf_params, signal_number, samples_per_image):

    cm = ['Greys'] if samples_per_image == 1 else ['Reds', 'Greens', 'Blues']

    figs = []
    for atom in range(H.shape[-2]):
        fig, axes = plt.subplots(nrows=1, ncols=samples_per_image, squeeze=False, figsize=(samples_per_image*10, 5),
                                 subplot_kw=dict(xticks=[], yticks=[], frame_on=False))

        for channel, ax in zip(range(samples_per_image), axes.flatten()):
            if nmf_params['shift_invariant']:
                im = ax.imshow(np.squeeze(H[..., atom, signal_number * samples_per_image + channel]), aspect='equal',
                               cmap=cm[channel])
            else:
                im = ax.imshow(H[0], aspect='equal')
            ax.set_title(f'atom {atom}, channel {channel}')
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        figs.append(fig)

    return figs


def st_show_activations(figs):
    st.markdown('## Activations')
    for fig in figs:
        st.pyplot(fig)


def plot_partial_reconstruction(nmf, nmf_params, signal_number, samples_per_image):
    # TODO: should be possible to merge this with plot_activations()
    cm = ['Greys'] if samples_per_image == 1 else ['Reds', 'Greens', 'Blues']

    figs = []
    for atom in range(nmf.n_components):
        fig, axes = plt.subplots(nrows=1, ncols=samples_per_image, squeeze=False, figsize=(samples_per_image*10, 5))

        for channel, ax in zip(range(samples_per_image), axes.flatten()):
            if nmf_params['shift_invariant']:
                H_partial = nmf.partial_reconstruct([atom, ], [signal_number * samples_per_image + channel, ])
                im = ax.imshow(np.squeeze(H_partial), aspect='equal', cmap=cm[channel])
            else:
                raise NotImplementedError
            ax.set_title(f'atom {atom}, channel {channel}')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax)

        figs.append(fig)

    return figs


def st_show_partial_reconstruction(figs):
    st.markdown('## Partial Reconstructions')

    col_def = [1] + [8] * (len(figs[0]) - 1)

    for atom_figs in figs:
        cols = st.beta_columns(col_def)
        for fig, col in zip(atom_figs, cols):
            col.pyplot(fig)


def close_figs(figs):
    if isinstance(figs, list):
        for fig in figs:
            close_figs(fig)
    else:
        plt.close(figs)


if __name__ == '__main__':
    # -------------------- settings -------------------- #

    st.sidebar.markdown('# General settings')

    auto_update = st.sidebar.checkbox('Auto-Update', False)
    force_refresh = st.sidebar.button('Refresh')
    seed = st.sidebar.number_input('Random seed', value=42)
    np.random.seed(seed)

    d, f, max_images, color_mode = st_define_dataset_params()
    images, image_shape = load_images(d, f, max_images, color_mode)

    nmf_params = st_define_nmf_params(image_shape)

    logging.info(f'NMF params: {nmf_params}')

    if not (auto_update or force_refresh):
        st.info('Auto-Update disabled')
        st.stop()

    # -------------------- model fitting -------------------- #

    # fit the NMF model
    nmf = deepcopy(compute_nmf(images, nmf_params))

    # -------------------- visualization -------------------- #

    # show learned dictionary
    dictionary_fig = plot_dictionary(nmf.W)
    st_show_dictionary(dictionary_fig)
    close_figs(dictionary_fig)

    st.markdown('# Signal reconstruction')

    # select signal to be visualized
    samples_per_image = 3 if color_mode == 'colors (identical basis)' else 1

    max_value = nmf.n_signals // samples_per_image - 1

    signal_number = 0 if nmf.n_signals == 1 else st.slider('Image number', min_value=0, max_value=max_value, value=0)

    # show reconstruction of individual signals
    signal_figs = plot_signal_reconstruction(nmf, signal_number, samples_per_image)
    st_show_signal_reconstruction(signal_figs)
    close_figs(signal_figs)

    # show activation pattern
    activation_figs = plot_activations(nmf.H, nmf_params, signal_number, samples_per_image)
    st_show_activations(activation_figs)
    close_figs(activation_figs)

    # show partial reconstructions
    # reconstruction_figs = plot_partial_reconstruction(nmf, nmf_params, signal_number, samples_per_image)
    # st_show_partial_reconstruction(reconstruction_figs)
    # close_figs(reconstruction_figs)
