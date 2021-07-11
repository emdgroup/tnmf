"""Provides functionality to generate certain 1-D and 2-D test signals."""

# pylint: disable=redefined-outer-name, unnecessary-lambda

from itertools import product, zip_longest
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from more_itertools import chunked
from numpy.linalg import norm


def generate_pulse(shape: str, length: int = 20) -> np.ndarray:
    """
    Generates a short signal pulse of specified shape and length.

    Parameters
    ----------
    shape : 'n' | '-' | '^' | 'v' | '_'
        Shape of the pulse.
    length : int
        Length of the pulse.

    Returns
    -------
    pulse : np.ndarray
        The signal pulse as a 1-D array.
    """
    # used for ramp-shaped pulses
    l1, l2 = np.ceil(length / 2), np.floor(length / 2)

    # define the signals shape
    if shape == 'n':
        r = (length - 1) / 2
        f = lambda x: np.sqrt((r ** 2) - ((x - r) ** 2))  # noqa: E731

    elif shape == '-':
        f = lambda x: np.ones_like(x, dtype=float)  # noqa: E731

    elif shape == '^':
        f = lambda x: np.hstack([np.arange(l1), l1 - 1 - (l1 != l2) - np.arange(l2)])  # noqa: E731

    elif shape == 'v':
        f = lambda x: np.hstack([l1 - 1 - np.arange(l2), np.arange(l1)])  # noqa: E731

    elif shape == '_':
        f = lambda x: np.zeros_like(x)  # noqa: E731

    else:
        raise ValueError('unknown pulse shape')

    # create the pulse
    x = np.arange(length)
    pulse = f(x)

    # normalize the signal
    if shape != '_':
        pulse /= norm(pulse)

    return pulse


def generate_pulse_train(
        symbols: Optional[List[str]] = None,
        pulse_length: int = 20,
        n_pulses: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a signal composed of a random sequence of multi-channel pulses.

    Parameters
    ----------
    symbols : List[str], optional
        A list of symbols (= multi-channel pulse shapes) specified via strings that are used as a dictionary to generate
        the signal. If 'None', the following three-channel symbols are used: ['nnn', '---', '^^^', 'vvv', '___'].
        See `generate_pulse` for an overview of all available pulse shapes.
    pulse_length : int
        The length of each individual pulse.
    n_pulses : int
        The total number of pulses to be sequenced.

    Returns
    -------
    signal : np.ndarray
        A 2-D array of shape (n_channels, n_pulses * pulse_length) containing the signal.
    W : np.ndarray
        A 3-D array of shape (n_symbols, n_channels, pulse_length) representing the pulse shape dictionary.
    """
    # default pulse shapes
    if symbols is None:
        symbols = ['nnn', '---', '^^^', 'vvv', '___']

    # assert that all provided symbols have the same number of channels
    else:
        if len(np.unique([len(p) for p in symbols])) != 1:
            raise ValueError('all symbols must have the same number of channels')

    # generate all pulse shapes
    W = np.stack([np.stack([generate_pulse(shape, pulse_length) for shape in symbol]) for symbol in symbols])

    # generate a random sequence of pulse indices and synthesize the signal
    sequence = np.random.choice(range(len(symbols)), n_pulses)
    signal = np.hstack([W[symbol_idx] for symbol_idx in sequence])

    return signal, W


def generate_patch(pattern: str, size: int = 10, color: Optional[str] = None) -> np.ndarray:
    """
    Generates a square image patch showing a certain pattern of specified size and color.

    Parameters
    ----------
    pattern : 'x' | '+' | 's'
        Pattern shown in the image patch.
    size : int
        Size of both dimensions of the image patch.
    color : 'r' | 'g' | 'b' | 'y' | 'm' | 'c' | 'w' , optional
        Color of the pattern. If 'None', the generated image patch will be grayscale.

    Returns
    -------
    patch : np.ndarray
        The image patch as a 3-D array, where the first dimension indexes the color channel.
    """
    # generate the grayscale version of the image patch
    if pattern == 'x':
        im = np.eye(size)
        im[np.rot90(im).astype(bool)] = 1
    elif pattern == '+':
        im = np.zeros([size, size])
        idx = np.array([np.floor((size-1)/2), np.ceil((size-1)/2)]).astype(int)
        im[idx, :] = 1
        im[:, idx] = 1
    elif pattern == 's':
        fill_width = int(size/3)
        square_width = size - 2 * fill_width
        im = np.pad(np.ones([square_width, square_width]), fill_width)
    else:
        raise ValueError('unknown patch shape')

    # convert the grayscale image to RGB
    if color:
        patch = np.zeros([3, size, size])
        color_dict = {'r': [0], 'g': [1], 'b': [2], 'y': [0, 1], 'm': [0, 2], 'c': [1, 2], 'w': [0, 1, 2]}
        channels = color_dict[color]
        patch[channels] = np.tile(im[None], [len(channels), 1, 1])
    else:
        patch = im[None]

    return patch


def generate_block_image(
        symbols: Optional[List[str]] = None,
        symbol_size: int = 10,
        n_symbols: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a block-structured image composed of several image patches, each containing a single random pattern.

    Parameters
    ----------
    symbols : List[str], optional
        A list of symbols used as a dictionary to generate the image. Each symbol is either a one-character or
        two-character string, where the first character specifies the symbol shape and the optional second character
        specifies the symbol color. For example: 'x' creates a grayscale cross-shaped patch while 'sr' creates a red square-
        shaped patch. If 'None', a certain default set of colored patches will be used as dictionary. See `generate_patch`
        for an overview of all available patch shapes and color options.
    symbol_size : int
        The size of both dimensions of each individual image patch.
    n_symbols : int
        The number of image patches to be stacked both horizontally and vertically.

    Returns
    -------
    image : np.ndarray
        A 3-D array of shape (3, n_symbols * symbol_size, n_symbols * symbol_size) containing the image.
    W : np.ndarray
        A 3-D array of shape (3, symbols_size, symbol_size) containing the image patch dictionary.
    """
    # default symbols
    if symbols is None:
        shapes = ['+', 'x', 's']
        colors = ['r', 'g', 'b', 'y', 'm', 'c', 'w']
        symbols = [''.join(spec) for spec in product(shapes, colors)]

    # separate shape and color information if colors are specified
    if len(symbols[0]) == 1:
        symbols = list(symbols).copy()
        symbols[0] = (symbols[0], None)
    shapes, colors = zip_longest(*symbols)

    # generate a random sequence of patches indices
    sequence = np.random.choice(range(len(symbols)), n_symbols ** 2)

    # generate all patch types
    W = np.stack([generate_patch(shape, symbol_size, color) for shape, color in zip(shapes, colors)])

    # turn the sequence of patch indices into a sequence patches and stack them into an image
    patches = [W[patch_idx] for patch_idx in sequence]
    image = np.block(list(chunked(patches, n_symbols)))

    return image, W


if __name__ == '__main__':

    plt.style.use('seaborn')

    # ---------- 1-D example ---------- #

    # specify pulse properties
    n_pulses = 6
    pulse_length = 100

    # generate the pulse signal
    signal, _ = generate_pulse_train(pulse_length=pulse_length, n_pulses=n_pulses)

    # visualize the signal and highlight the individual pulses
    fig, axs = plt.subplots(nrows=signal.shape[1])
    for channel, ax in enumerate(axs):
        for p in range(n_pulses):
            x = range(p * pulse_length, p * pulse_length + pulse_length)
            ax.plot(x, signal[x, channel])
    plt.show()

    # ---------- 2-D example ---------- #

    # specify image patch properties
    symbol_size = 11
    n_symbols = 10

    # generate and visualize the image
    im, _ = generate_block_image(symbol_size=symbol_size, n_symbols=n_symbols)
    plt.imshow(im)
    plt.axis('off')
    plt.grid(False)
    plt.tight_layout()
    plt.show()
