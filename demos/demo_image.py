import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from tnmf.TransformInvariantNMF import TransformInvariantNMF
from tnmf.utils.data_loading import racoon_image

progress_bar = st.sidebar.progress(0)

st.sidebar.markdown('# Input options')
scale = st.sidebar.slider('Image rescaling factor', min_value=0., max_value=1., value=0.2)
color = st.sidebar.checkbox('Color channels', False)

st.sidebar.markdown('# NMF options')
backend = st.sidebar.selectbox('Backend', ['numpy', 'numpy_fft', 'numpy_caching_fft', 'pytorch', 'pytorch_fft'], 1)
n_atoms = st.sidebar.number_input('Number of atoms', min_value=1, value=10)
atom_size = st.sidebar.number_input('Atom size', min_value=1, value=10)
sparsity_H = st.sidebar.number_input('Activation sparsity', min_value=0., value=0., step=0.1)
inhibition = st.sidebar.number_input('Activation lateral inhibition', min_value=0., value=0., step=0.1)
if inhibition > 0:
    inhibition_range = st.sidebar.selectbox('Inhibition Range', [f'default ({atom_size-1})', ] + list(range(1, atom_size)), 0)
    inhibition_range = None if isinstance(inhibition_range, str) else (inhibition_range, ) * 2
else:
    inhibition_range = None

n_iterations = st.sidebar.number_input('Number of iterations', min_value=1, value=100)

img = racoon_image(gray=not color, scale=scale)
V = img.transpose((2, 0, 1))[np.newaxis, ...] if color else img[np.newaxis, np.newaxis, ...]

nmf = TransformInvariantNMF(
    n_atoms=n_atoms,
    atom_shape=(atom_size, atom_size),
    n_iterations=n_iterations,
    backend=backend,
    inhibition_range=inhibition_range,
)

nmf.fit(
    V,
    sparsity_H=sparsity_H,
    inhibition_strength=inhibition,
    progress_callback=lambda _, x: progress_bar.progress((x + 1) / n_iterations)
)

img_cmap = None if color else 'gray'

st.markdown('# Input and reconstruction')
col1, col2 = st.beta_columns(2)
with col1:
    fig = plt.figure()
    plt.imshow(img, cmap=img_cmap)
    st.pyplot(fig)
with col2:
    fig = plt.figure()
    plt.imshow(nmf.R[0].transpose((1, 2, 0)), cmap=img_cmap)
    st.pyplot(fig)

st.markdown('## Atom, activation and partial reconstruction')
for i_atom in range(n_atoms):
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        fig = plt.figure()
        plt.imshow(nmf.W[i_atom, 0], cmap=img_cmap)
        st.pyplot(fig)
    with col2:
        fig = plt.figure()
        plt.imshow(nmf.H[0, i_atom])
        st.pyplot(fig)
    with col3:
        fig = plt.figure()
        partial_R = nmf.R_partial(i_atom)
        plt.imshow(partial_R[0].transpose((1, 2, 0)) / partial_R[0].max(), cmap=img_cmap)
        st.pyplot(fig)
