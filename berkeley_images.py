"""
Author: Mathias Winkel
"""

import os
import glob
import imageio
import logging

import numpy as np
from scipy import fftpack
import matplotlib
import matplotlib.pyplot as plt
from TransformInvariantNMF import SparseNMF, ShiftInvariantNMF

matplotlib.use('AGG')
plt.style.use('seaborn')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')


COLOR_SELECTIONS = {
    'grey': lambda img: [np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])[:, :, np.newaxis]],
    'red': lambda img: [img[..., 0:1]],
    'green': lambda img: [img[..., 1:2]],
    'blue': lambda img: [img[..., 2:3]],
    'color basis': lambda img: [img],
    'colors (identical basis)': lambda img: [img[..., 0:1], img[..., 1:2], img[..., 2:3]],
}


def filter_channel(ch):
    """
    combined low-pass/whitening filter as in

    Olshausen, B. A., & Field, D. J. (1997). Sparse coding with an overcomplete basis set: A strategy employed by V1?
    Vision Research, 37(23), 3311–3325. https://doi.org/10.1016/S0042-6989(97)00169-7
    :param ch:
    :return:
    """
    spectrum = fftpack.fft2(ch)

    f0x, f0y = ch.shape[0] // 2, ch.shape[1] // 2

    x = np.arange(-f0x, f0x, 1)
    y = np.arange(-f0y, f0y, 1)
    yy, xx = np.meshgrid(y, x)
    ff = np.sqrt((xx / f0x) ** 2 + (yy / f0y) ** 2)
    filter = ff * np.exp(-ff ** 4)

    spectrum *= filter[:, :, np.newaxis]

    return np.clip(fftpack.ifft2(spectrum), 0., 1.)


def load_images(path: str, pattern: str, max_images: int = 0, remove_margin=0, filter=False,
                color_mode: str = 'grey', dtype=np.float32) -> (np.ndarray, tuple):
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
            ch = channel[remove_margin:-remove_margin-1, remove_margin:-remove_margin-1, :]
            rows, cols = min(rows, ch.shape[0]), min(cols, ch.shape[1])
            images.append(filter_channel(ch) if filter else ch)

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


def compute_nmf(V, nmf_params, progress_callback):
    """Streamlit caching of NMF fitting."""
    nmf_params = nmf_params.copy()
    if nmf_params.pop('shift_invariant'):
        nmf = ShiftInvariantNMF(**nmf_params)
    else:
        nmf = SparseNMF(**nmf_params)
    nmf.fit(V, progress_callback)
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

    return [fig]


def plot_dictionary(W, num_columns=5):
    nrows = (W.shape[-1] + num_columns - 1) // num_columns
    fig = plt.figure(figsize=(2*num_columns, 2*nrows))

    for m in range(W.shape[-1]):
        ax = fig.add_subplot(nrows, num_columns, m+1, xticks=[], yticks=[])
        im = ax.imshow(W[..., m].squeeze())
        ax.set_title(f'{m}')
        ax.grid(False)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    return [fig]


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


def plot_partial_reconstruction(nmf, nmf_params, signal_number, samples_per_image):
    # TODO: should be possible to merge this with plot_activations()
    cm = ['Greys'] if samples_per_image == 1 else ['Reds', 'Greens', 'Blues']

    figs = []
    for atom in range(nmf.n_components):
        fig, axes = plt.subplots(nrows=1, ncols=samples_per_image, squeeze=False, figsize=(samples_per_image*10, 5))

        for channel, ax in zip(range(samples_per_image), axes.flatten()):
            if nmf_params['shift_invariant']:
                H_partial = nmf.partial_reconstruct(signal_number * samples_per_image + channel, 0, atom)
                im = ax.imshow(np.squeeze(H_partial), aspect='equal', cmap=cm[channel], vmin=0., vmax=1.)
            else:
                raise NotImplementedError
            ax.set_title(f'atom {atom}, channel {channel}')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax)

        figs.append(fig)

    return figs


def close_figs(figs):
    if isinstance(figs, list):
        for fig in figs:
            close_figs(fig)
    else:
        plt.close(figs)


def st_plot(title, figs):
    for ifig, fig in enumerate(figs):
        filename = f'{title}_{ifig}.png'
        logging.info(f'Saving {filename}')
        fig.savefig(filename)
    close_figs(figs)


if __name__ == '__main__':
    # -------------------- settings -------------------- #

    np.random.seed(42)

    dataset_params = {
        'path': r'BSR_bsds500/BSR/BSDS500/data/images/train',
        'pattern': '*.jpg',
        'max_images': 5,
        'remove_margin': 0,
        'color_mode': 'colors (identical basis)',
        'dtype': np.float64,
        'filter': False,
    }

    logging.info(f'dataset params: {dataset_params}')

    images, image_shape = load_images(**dataset_params)

    nmf_params = {
        'verbose': 2,
        'method': 'fftconvolve',
        'reconstruction_mode': 'full', # 'same', 'full', 'valid'
        'shift_invariant': True,
        'sparsity_H': 0.5,
        'n_iterations': 200,
        'refit_H': True,
        'n_components': 16,
        'atom_size': 9,
        'inhibition_range': None,
        'inhibition_strength': 0.25,
    }

    logging.info(f'NMF params: {nmf_params}')

    # -------------------- model fitting -------------------- #

    def progress_callback(i: int, **energy) -> bool:
        logging.info(f"Iteration: {i}\tError terms: {energy}")
        return True

    # fit the NMF model
    nmf = compute_nmf(images, nmf_params, progress_callback=progress_callback)

    # -------------------- visualization -------------------- #

    color_mode = dataset_params['color_mode']
    st_plot(f'Learned dictionary - {color_mode}', plot_dictionary(nmf.W))

    # select signal to be visualized
    samples_per_image = 3 if color_mode == 'colors (identical basis)' else 1

    for signal_number in range(nmf.n_signals//samples_per_image - 1):
        st_plot(f'Comparison to original signal - sample {signal_number:04d}',
                plot_signal_reconstruction(nmf, signal_number, samples_per_image))

        st_plot(f'Activations - sample {signal_number:04d}',
                plot_activations(nmf.H, nmf_params, signal_number, samples_per_image))

        st_plot(f'Partial Reconstructions - sample {signal_number:04d}',
                plot_partial_reconstruction(nmf, nmf_params, signal_number, samples_per_image))
