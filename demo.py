# TODO: move file to demo folder once everything is packaged

import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from tnmf.TransformInvariantNMF import TransformInvariantNMF
from tnmf.utils.data_loading import racoon_image

progress_bar = st.sidebar.progress(0)

st.sidebar.markdown('# Input options')
scale = st.sidebar.slider('Image rescaling factor', min_value=0., max_value=1., value=0.2)
color = st.sidebar.checkbox('Color channels', False)

st.sidebar.markdown('# NMF options')
backend = st.sidebar.selectbox('Backend', ['numpy', 'numpy_fft', 'pytorch'], 1)
n_atoms = st.sidebar.number_input('Number of atoms', min_value=1, value=10)
atom_size = st.sidebar.number_input('Atom size', min_value=1, value=10)
sparsity_H = st.sidebar.number_input('Activation sparsity', min_value=0., value=0., step=0.1)
n_iterations = st.sidebar.number_input('Number of iterations', min_value=1, value=100)

img = racoon_image(gray=not color, scale=scale)
V = img.transpose((2, 0, 1))[np.newaxis, ...] if color else img[np.newaxis, np.newaxis, ...]

nmf = TransformInvariantNMF(
    n_atoms=n_atoms,
    atom_shape=(atom_size, atom_size),
    n_iterations=n_iterations,
    backend=backend,
)

nmf.fit(
    V,
    sparsity_H=sparsity_H,
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
        # TODO: add partial reconstruction function
        fig = plt.figure()
        partial_R = nmf._backend.reconstruct(nmf._W[i_atom:i_atom+1], nmf._H[:1, i_atom:i_atom+1])
        if torch.is_tensor(partial_R):
            partial_R = partial_R.detach().numpy()
        plt.imshow(partial_R[0].transpose((1, 2, 0)) / partial_R[0].max(), cmap=img_cmap)
        st.pyplot(fig)
