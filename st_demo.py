"""
Author: Adrian Sosic
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from signals import generate_pulse_train, generate_pulse
from TransformInvariantNMF import SparseNMF, ShiftInvariantNMF
from itertools import zip_longest
from copy import deepcopy


@st.cache
def compute_nmf(V, nmf_params, shift_invariant):
	"""Streamlit caching of NMF fitting."""
	if shift_invariant:
		nmf = ShiftInvariantNMF(**nmf_params)
	else:
		nmf = SparseNMF(**nmf_params)
	nmf.fit(V)
	return nmf


def plot_signal_matrix(V):
	"""Plots a given signal matrix as image."""
	fig = plt.figure(figsize=(5, 5))
	plt.imshow(V, cmap='plasma', aspect='auto')
	plt.xlabel('signal number')
	plt.ylabel('signal dimension')
	plt.xticks([])
	plt.yticks([])
	return fig


# -------------------- settings -------------------- #

# fix random seed
np.random.seed(0)

# define signal parameters
st.sidebar.markdown('# Signal settings')
n_signals = st.sidebar.number_input('# Signals', min_value=1, value=100)
n_pulses = st.sidebar.number_input('# Pulses', min_value=1, value=3)
shapes = st.sidebar.multiselect('Pulse shapes', ['n', '-', '^', 'v', '_'], ['n', '-', '^', 'v', '_'])
pulse_length = st.sidebar.number_input('Pulse length', min_value=1, value=20)

# create signal matrix
V = np.array([generate_pulse_train(shapes=shapes, pulse_length=pulse_length, n_pulses=n_pulses)
			  for _ in range(n_signals)]).T

# define NMF parameters
st.sidebar.markdown('# NMF settings')
nmf_params = dict(
	sparsity_H=st.sidebar.number_input('Activation sparsity', min_value=0.0, value=0.1),
	n_iterations=st.sidebar.number_input('# Iterations', min_value=1, value=100),
	refit_H=st.sidebar.checkbox('Refit activations without sparsity', True)
)
shift_invariant = st.sidebar.checkbox('Shift invariant', True)
if shift_invariant:
	nmf_params['atom_size'] = st.sidebar.number_input('Atom size', min_value=0, max_value=V.shape[0], value=pulse_length)
	inhibition_str = f"Auto ({nmf_params['atom_size']})"
	inhibition = st.sidebar.radio('Inhbition range', [inhibition_str, 'Manual'], 0)
	if inhibition == inhibition_str:
		nmf_params['inhibition_range'] = None
	else:
		nmf_params['inhibition_range'] = st.sidebar.number_input('Inhibition range', min_value=0, value=nmf_params['atom_size'])
	nmf_params['inhibition_strength'] = st.sidebar.number_input('Inhibition strength', min_value=0.0, value=0.1)
	n_complete = len(shapes)
else:
	n_complete = len(shapes) * n_pulses
n_complete_str = f'complete ({n_complete} atoms)'
dict_size = st.sidebar.radio('Dictionary size', [n_complete_str, 'manual'], 0)
if dict_size == n_complete_str:
	n_components = n_complete
else:
	n_components = st.sidebar.number_input('# Dictionary elements', min_value=1, value=len(shapes) * n_pulses)
nmf_params['n_components'] = n_components


# -------------------- model fitting -------------------- #

# fit the NMF model
nmf = deepcopy(compute_nmf(V, nmf_params, shift_invariant))


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
s = st.slider('Signal number', min_value=0, max_value=n_signals-1, value=0)
fig = plt.figure()
plt.plot(V[:, s], label='signal')
plt.plot(nmf.R[:, s], label='reconstruction', color='tab:red')
plt.legend()
st.pyplot(fig)

# show ground truth pulses
st.markdown('# Ground truth signal pulses')
fig, axs = plt.subplots(figsize=(10, 2), ncols=len(shapes))
pulses = [generate_pulse(shape, length=pulse_length) for shape in shapes]
for ax, pulse, shape in zip_longest(axs.ravel(), pulses, shapes):
	if pulse is not None:
		ax.plot(pulse)
	if shape == '_':
		ax.set_ylim([-0.05, 1])
	ax.axis('off')
st.pyplot(fig)

# show dictionary
st.markdown('# Learned dictionary')
fig = nmf.plot_dictionary()
st.pyplot(fig)

# show activation pattern
st.markdown('# Activation pattern')
fig = plt.figure(figsize=(10, 5))
if shift_invariant:
	plt.imshow(nmf.H[:, :, s], aspect='auto')
	plt.xlabel('atom number')
	plt.ylabel('transformation number')
else:
	plt.imshow(nmf.H[0], aspect='auto')
	plt.xlabel('signal number')
	plt.ylabel('atom number')
plt.colorbar()
st.pyplot(fig)
