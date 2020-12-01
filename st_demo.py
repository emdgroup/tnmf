"""
Author: Adrian Sosic
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from signals import generate_pulse_train
from TransformInvariantNMF import SparseNMF, ShiftInvariantNMF
from itertools import product, repeat
from more_itertools import chunked
from copy import deepcopy


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


def plot_signal_matrix(V):
	"""Plots a given signal matrix as image."""
	fig = plt.figure(figsize=(5, 5))
	plt.imshow(np.reshape(V.transpose([1, 0, 2]), [-1, V.shape[2]]), cmap='plasma', aspect='auto')
	plt.xlabel('signal number')
	plt.ylabel('signal dimension')
	plt.xticks([])
	plt.yticks([])
	return fig


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

	return signal_params


def create_signal(signal_type, signal_params):
	pulse_params = {k: signal_params[k] for k in ['symbols', 'pulse_length', 'n_pulses']}
	V = np.array([generate_pulse_train(**pulse_params) for _ in range(signal_params['n_signals'])]).T
	return V


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
		complete_dict_size = len(signal_params['symbols']) * signal_params['n_pulses']

	n_complete_str = f'complete ({complete_dict_size} atoms)'
	dict_size = st.sidebar.radio('Dictionary size', [n_complete_str, 'manual'], 0)
	if dict_size == n_complete_str:
		n_components = complete_dict_size
	else:
		n_components = st.sidebar.number_input('# Dictionary elements', min_value=1, value=len(signal_params['symbols']) * signal_params['n_pulses'])

	nmf_params.update(dict(
		n_components=n_components,
	))

	# -------------------- settings for shift invariance -------------------- #

	if nmf_params['shift_invariant']:

		st.sidebar.markdown('### Shift invariance settings')
		atom_size = st.sidebar.number_input('Atom size', min_value=0, max_value=V.shape[0], value=signal_params['pulse_length'])
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


# -------------------- settings -------------------- #

st.sidebar.markdown('# General settings')

# fix random seed
seed = st.sidebar.number_input('Random seed', value=42)
np.random.seed(seed)

# define signal type
signal_type = st.sidebar.radio('Signal type', ['Time series'], 0)

# define signal parameters
signal_params = st_define_signal_params(signal_type)

# create signal tensor
V = create_signal(signal_type, signal_params)

# define NMF parameters
nmf_params = st_define_nmf_params(signal_type, signal_params)


# -------------------- model fitting -------------------- #

# fit the NMF model
nmf = deepcopy(compute_nmf(V, nmf_params))


# -------------------- visualization -------------------- #

# show global reconstruction
st.markdown('# Signal reconstruction')
col1, col2 = st.beta_columns(2)
with col1:
	fig = plot_signal_matrix(V)
	plt.title('signal matrix')
	st.pyplot(fig)
with col2:
	fig = plot_signal_matrix(nmf.R)
	plt.title('reconstruction')
	st.pyplot(fig)

# show reconstruction of individual signals
s = st.slider('Signal number', min_value=0, max_value=signal_params['n_signals']-1, value=0)
fig, axs = plt.subplots(nrows=nmf.n_channels, ncols=1)
axs = np.atleast_1d(axs)
for channel, ax in enumerate(axs):
	ax.plot(V[:, channel, s], label='signal')
	ax.plot(nmf.R[:, channel, s], label='reconstruction', color='tab:red')
plt.legend()
st.pyplot(fig)

# show ground truth pulses
st.markdown('# Ground truth signal pulses')
W = np.array([generate_pulse_train([symbol], n_pulses=1, pulse_length=signal_params['pulse_length']) for symbol in signal_params['symbols']]).T
figs = nmf.plot_dictionary(W)
for row in chunked(figs, 3):
	cols = st.beta_columns(3)
	for fig, col in zip(row, cols):
		with col:
			st.pyplot(fig)

# show dictionary
st.markdown('# Learned dictionary')
figs = nmf.plot_dictionary()
for row in chunked(figs, 3):
	cols = st.beta_columns(3)
	for fig, col in zip(row, cols):
		with col:
			st.pyplot(fig)

# show activation pattern
st.markdown('# Activation pattern')
fig = plt.figure(figsize=(10, 5))
if nmf_params['shift_invariant']:
	plt.imshow(nmf.H[:, :, s], aspect='auto')
	plt.xlabel('atom number')
	plt.ylabel('transformation number')
else:
	plt.imshow(nmf.H[0], aspect='auto')
	plt.xlabel('signal number')
	plt.ylabel('atom number')
plt.colorbar()
st.pyplot(fig)
