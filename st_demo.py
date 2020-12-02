"""
Author: Adrian Sosic
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from signals import generate_pulse_train, generate_block_image
from TransformInvariantNMF import SparseNMF, ShiftInvariantNMF
from itertools import product, repeat
from more_itertools import chunked
from copy import deepcopy

# TODO: get available signal parameter options (shapes, colors, ...) directly from signals file
# TODO: implement wrapper/transformation to handle shift-dimensions in TransformInvariantNMF


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


def st_define_signal_params(signal_type):

	st.sidebar.markdown('# Signal settings')

	if signal_type == 'Time series':

		signal_params = dict(
			n_signals=st.sidebar.number_input('# Signals', min_value=1, value=100),
			n_pulses=st.sidebar.number_input('# Pulses', min_value=1, value=3),
			n_channels=st.sidebar.number_input('# Channels', min_value=1, value=3, max_value=5),
		)

		shapes = ['n', '-', '^', 'v', '_']
		symbols = [''.join(chars) for chars in product(*repeat(shapes, signal_params['n_channels']))]

		signal_params.update(dict(
			symbols=st.sidebar.multiselect('Symbols', symbols, np.array(symbols)[np.random.randint(0, len(symbols), 5)]),
			pulse_length=st.sidebar.number_input('Pulse length', min_value=1, value=20)
		))

	elif signal_type == 'Image':

		signal_params = dict(
			n_signals=st.sidebar.number_input('# Signals', min_value=1, value=1),
			n_symbols=st.sidebar.number_input('# Patches', min_value=1, value=5),
			n_channels=st.sidebar.radio('# Channels', [1, 3], 0),
			symbol_size=st.sidebar.number_input('# Patch size', min_value=1, value=10),
		)

		shapes = ['+', 'x', 's']
		colors = ['r', 'g', 'b', 'y', 'm', 'c', 'w'] if signal_params['n_channels'] == 3 else ['']
		symbols = [''.join(spec) for spec in product(shapes, colors)]

		signal_params.update(dict(
			symbols=st.sidebar.multiselect('Symbols', symbols, symbols)
		))

	return signal_params


def generate_signal(signal_type, signal_params):

	if signal_type == 'Time series':
		atom_params = {k: signal_params[k] for k in ['symbols', 'pulse_length', 'n_pulses']}
		signal_generator = generate_pulse_train
	elif signal_type == 'Image':
		atom_params = {k: signal_params[k] for k in ['symbols', 'symbol_size', 'n_symbols']}
		signal_generator = generate_block_image

	signals = []
	for _ in range(signal_params['n_signals']):
		s, W = signal_generator(**atom_params)
		signals.append(s)
	V = np.stack(signals, axis=-1)

	return V, W


def st_define_nmf_params(signal_type, signal_params):

	st.sidebar.markdown('# NMF settings')

	# -------------------- general settings -------------------- #

	nmf_params = dict(
		shift_invariant=st.sidebar.checkbox('Shift invariant', True),
		sparsity_H=st.sidebar.number_input('Activation sparsity', min_value=0.0, value=0.1),
		n_iterations=st.sidebar.number_input('# Iterations', min_value=1, value=100),
		refit_H=st.sidebar.checkbox('Refit activations without sparsity', True)
	)

	# -------------------- dictionary size  -------------------- #

	if nmf_params['shift_invariant']:
		complete_dict_size = len(signal_params['symbols'])
	else:
		if signal_type == 'Time series':
			complete_dict_size = len(signal_params['symbols']) * signal_params['n_pulses']
		elif signal_type == 'Image':
			complete_dict_size = len(signal_params['symbols']) * (signal_params['n_symbols'] ** 2)

	n_complete_str = f'complete ({complete_dict_size} atoms)'
	dict_size = st.sidebar.radio('Dictionary size', [n_complete_str, 'manual'], 0)
	if dict_size == n_complete_str:
		n_components = complete_dict_size
	else:
		n_components = st.sidebar.number_input('# Dictionary elements', min_value=1, value=complete_dict_size)

	nmf_params.update(dict(
		n_components=n_components,
	))

	# -------------------- settings for shift invariance -------------------- #

	if nmf_params['shift_invariant']:

		st.sidebar.markdown('### Shift invariance settings')
		if signal_type == 'Time series':
			default_atom_size = signal_params['pulse_length']
		elif signal_type == 'Image':
			default_atom_size = signal_params['symbol_size']
		atom_size = st.sidebar.number_input('Atom size', min_value=0, max_value=V.shape[0], value=default_atom_size)
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


def st_compare_signals(nmf):

	def plot_signal_matrix(V):
		fig = plt.figure(figsize=(5, 5))
		plt.imshow(np.reshape(V, [-1, V.shape[-1]], order='F'), cmap='plasma', aspect='auto')
		plt.xlabel('signal number')
		plt.ylabel('signal dimension')
		plt.xticks([])
		plt.yticks([])
		return fig

	st.markdown('# Global signal reconstruction')
	col1, col2 = st.beta_columns(2)
	with col1:
		fig = plot_signal_matrix(V)
		plt.title('signal matrix')
		st.pyplot(fig)
	with col2:
		fig = plot_signal_matrix(nmf.R)
		plt.title('reconstruction')
		st.pyplot(fig)


def st_compare_single_signals(nmf, signal_type, signal_number):

	st.markdown('# Signal reconstruction')

	if signal_type == 'Time series':
		fig, axs = plt.subplots(nrows=nmf.n_channels, ncols=1)
		axs = np.atleast_1d(axs)
		for channel, ax in enumerate(axs):
			ax.plot(V[:, channel, signal_number], label='signal')
			ax.plot(nmf.R[:, channel, signal_number], label='reconstruction', color='tab:red')
		plt.legend()
		st.pyplot(fig)

	elif signal_type == 'Image':
		col1, col2 = st.beta_columns(2)
		with col1:
			fig = plt.figure()
			plt.imshow(nmf.V[..., signal_number])
			plt.grid(False)
			st.pyplot(fig)
		with col2:
			fig = plt.figure()
			plt.imshow(nmf.R[..., signal_number])
			plt.grid(False)
			st.pyplot(fig)


def compare_dictionaries(W_true, W_learned, signal_type):

	def plot_dictionary(W):
		figs = []
		for m in range(W.shape[-1]):
			if signal_type == 'Time series':
				fig, axs = plt.subplots(nrows=W.shape[1], ncols=1)
				for w, ax in zip(W[:, :, m].T, axs):
					ax.plot(w)
					if np.allclose(w, 0):
						ax.set_ylim([-0.05, 1])
			elif signal_type == 'Image':
				fig = plt.figure()
				plt.imshow(W[..., m])
				plt.grid(False)
				plt.axis('off')
			plt.tight_layout()
			figs.append(fig)
		return figs

	def st_show_dictionary(W):
		figs = plot_dictionary(W)
		for row in chunked(figs, 3):
			cols = st.beta_columns(3)
			for fig, col in zip(row, cols):
				with col:
					st.pyplot(fig)

	st.markdown('# Ground truth dictionary')
	st_show_dictionary(W_true)
	st.markdown('# Learned dictionary')
	st_show_dictionary(W_learned)


def st_plot_activations(H, nmf_params, signal_type, signal_number):

	st.markdown('# Activation pattern')

	m = st.slider('Dictionary atom', min_value=0, max_value=H.shape[-2]-1)
	fig = plt.figure(figsize=(10, 5))
	if nmf_params['shift_invariant']:
		if signal_type == 'Time series':
			plt.imshow(H[..., signal_number], aspect='auto')
		elif signal_type == 'Image':
			plt.imshow(H[..., m, signal_number], aspect='auto')
	else:
		plt.imshow(H[0], aspect='auto')
	plt.colorbar()
	plt.grid(False)
	st.pyplot(fig)


if __name__ == '__main__':

	# -------------------- settings -------------------- #

	st.sidebar.markdown('# General settings')

	# fix random seed
	seed = st.sidebar.number_input('Random seed', value=42)
	np.random.seed(seed)

	# define signal type
	signal_type = st.sidebar.radio('Signal type', ['Time series', 'Image'], 1)

	# define signal parameters
	signal_params = st_define_signal_params(signal_type)

	# create signal tensor
	V, W = generate_signal(signal_type, signal_params)

	# define NMF parameters
	nmf_params = st_define_nmf_params(signal_type, signal_params)


	# -------------------- model fitting -------------------- #

	# fit the NMF model
	nmf = deepcopy(compute_nmf(V, nmf_params))


	# -------------------- visualization -------------------- #

	# show global reconstruction
	st_compare_signals(nmf)

	# select signal to be visualized
	signal_number = 0 if nmf.n_signals == 1 else st.slider('Signal number', min_value=0, max_value=nmf.n_signals-1, value=0)

	# show reconstruction of individual signals
	st_compare_single_signals(nmf, signal_type, signal_number)

	# show activation pattern
	st_plot_activations(nmf.H, nmf_params, signal_type, signal_number)

	# show ground truth pulses
	compare_dictionaries(W, nmf.W, signal_type)
