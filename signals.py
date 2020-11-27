"""
Author: Adrian Sosic
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from typing import Optional, List
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


if __name__ == '__main__':

	# ---------- demo example ---------- #

	# generate pulse signal
	n_pulses = 6
	pulse_length = 100
	symbols = ['v^v', 'n-n']
	signal = generate_pulse_train(symbols=symbols, pulse_length=pulse_length, n_pulses=n_pulses)

	# visualize the signal and highlight the individual pulses
	fig, axs = plt.subplots(nrows=signal.shape[0], ncols=1)
	for channel, ax in enumerate(axs):
		for p in range(n_pulses):
			x = range(p * pulse_length, p * pulse_length + pulse_length)
			ax.plot(x, signal[channel, x])
	plt.show()
