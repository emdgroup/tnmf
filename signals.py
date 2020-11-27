"""
Author: Adrian Sosic
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from typing import Optional, List
from itertools import product, repeat
from more_itertools import chunked
plt.style.use('seaborn')


def generate_pulse(shape: str, length: int = 20) -> np.array:
	"""
	Generates a short signal pulse of specified length and shape.

	Parameters
	----------
	shape : 'n' | '-' | '^' | 'v' | '_'
		Shape of the pulse.
	length : int
		Length of the pulse.

	Returns
	-------
	The signal pulse as a 1-D numpy array.
	"""

	# used for ramp-shaped pulses
	l1, l2 = np.ceil(length / 2), np.floor(length / 2)

	# define the signals shape
	if shape == 'n':
		r = (length - 1) / 2
		f = lambda x: np.sqrt((r ** 2) - ((x - r) ** 2))

	elif shape == '-':
		f = lambda x: np.ones_like(x)

	elif shape == '^':
		f = lambda x: np.hstack([np.arange(l1), l1 - 1 - (l1 != l2) - np.arange(l2)])

	elif shape == 'v':
		f = lambda x: np.hstack([l1 - 1 - np.arange(l2), np.arange(l1)])

	elif shape == '_':
		f = lambda x: np.zeros_like(x)

	else:
		raise ValueError('unknown shape')

	# create the pulse
	x = np.arange(length)
	pulse = f(x)

	# normalize the signal
	if shape != '_':
		pulse = pulse / norm(pulse)

	return pulse


def generate_pulse_train(
		symbols: Optional[List[str]] = None,
		pulse_length: int = 20,
		n_pulses: int = 5) -> np.array:
	"""
	Generates a signal composed of a random sequence of multi-channel pulses.

	Parameters
	----------
	symbols : (optional) List[str]
		A list of symbols (=multi-channel pulse shapes) specified via strings that are used as a dictionary to generate
		the signal. If None, the following three-channel symbols are used: ['nnn', '---', '^^^', 'vvv', '___'].
	pulse_length : int
		The length of each individual pulse.
	n_pulses : int
		The total number of pulses to be sequenced.

	Returns
	-------
	The signal as a 2-D numpy array, where the first dimension indexes the signal channel.
	"""
	# default pulse shapes
	if symbols is None:
		symbols = ['nnn', '---', '^^^', 'vvv', '___']

	# generate a random sequence of pulses and synthesize the signal
	sequence = np.random.choice(symbols, n_pulses)
	signal = np.hstack([np.vstack([generate_pulse(shape, pulse_length)
								   for shape in symbol])
						for symbol in sequence])

	return signal


def generate_patch(pattern: str, size: int = 10, color: Optional[str] = None) -> np.array:
	"""
	Generates a square image patch showing a certain pattern of specified size and color.

	Parameters
	----------
	pattern : 'x' | '+'
		Pattern shown in the image patch.
	size : int
		Size of both dimensions of the image patch.
	color : (optional) 'r' | 'g' | 'b' | 'y' | 'm' | 'c' | 'w'
		Color of the pattern.

	Returns
	-------
	If a 'color' is provided: the image patch as a 3-D numpy array, where the last dimension indexes the color channel.
	If no 'color' is provided: the image patch as a 2-D numpy array.
	"""

	# generate the grayscale image
	if pattern == 'x':
		im = np.eye(size)
		im[np.rot90(im).astype(bool)] = 1

	elif pattern == '+':
		im = np.zeros([size, size])
		idx = np.array([np.floor((size-1)/2), np.ceil((size-1)/2)]).astype(int)
		im[idx, :] = 1
		im[:, idx] = 1

	else:
		raise ValueError('unknown shape')

	# convert the grayscale image to RGB
	if color:
		im_rgb = np.zeros([size, size, 3])
		color_dict = {'r': [0], 'g': [1], 'b': [2], 'y': [0, 1], 'm': [0, 2], 'c': [1, 2], 'w': [0, 1, 2]}
		channels = color_dict[color]
		im_rgb[:, :, channels] = np.tile(im[:, :, None], [1, 1, len(channels)])
		im = im_rgb

	return im


def generate_block_image(
		symbols: Optional[List[str]] = None,
		symbol_size: int = 10,
		n_symbols: int = 10) -> np.array:
	"""
	Generates an image composed of several image patches, each containing a single random pattern.

	Parameters
	----------
	symbols : (optional) List[str]
		A list of symbols used as a dictionary to generate the image. Each symbol is either a one-character or
		two-character string, where the first character specifies the symbol shape and the optional second character
		specifies the symbol color.
	symbol_size : int
		The size of both dimensions of each individual image patch.
	n_symbols : int
		The number of image patches to be stacked both horizontally and vertically.

	Returns
	-------
	The image as a 2-D or 3-D numpy array, depending on whether or not the provided symbols contain color information.
	If color is provided, the last dimension indexes the color channel.
	"""
	# default symbols
	if symbols is None:
		shapes = ['+', 'x']
		colors = ['r', 'g', 'b', 'y', 'm', 'c', 'w']
		symbols = [''.join(spec) for spec in product(shapes, colors)]

	# generate a random sequence of patches
	sequence = np.random.choice(symbols, n_symbols ** 2)

	# separate shape and color information if colors are specified
	try:
		shapes, colors = zip(*sequence)
	except ValueError:
		shapes = sequence
		colors = repeat(None)

	# turn the sequence of symbols into a list patches and stack them into an image
	patches = [generate_patch(shape, symbol_size, color).T for shape, color in zip(shapes, colors)]
	image = np.block(list(chunked(patches, n_symbols))).T

	return image


if __name__ == '__main__':

	# ---------- 1-D example ---------- #

	# specify pulse properties
	n_pulses = 6
	pulse_length = 100

	# generate the pulse signal
	signal = generate_pulse_train(symbols=None, pulse_length=pulse_length, n_pulses=n_pulses)

	# visualize the signal and highlight the individual pulses
	fig, axs = plt.subplots(nrows=signal.shape[0], ncols=1)
	for channel, ax in enumerate(axs):
		for p in range(n_pulses):
			x = range(p * pulse_length, p * pulse_length + pulse_length)
			ax.plot(x, signal[channel, x])
	plt.show()

	# ---------- 2-D example ---------- #

	# specify image patch properties
	symbol_size = 11
	n_symbols = 10

	# generate and visualize the image
	im = generate_block_image(symbols=None, symbol_size=symbol_size, n_symbols=n_symbols)
	plt.imshow(im), plt.axis('off'), plt.grid(False), plt.tight_layout(), plt.show()
