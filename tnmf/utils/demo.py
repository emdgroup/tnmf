from abc import ABC, abstractmethod
from itertools import product, repeat
from typing import List, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from tnmf.TransformInvariantNMF import TransformInvariantNMF
from tnmf.utils.signals import generate_pulse_train, generate_block_image


def st_define_nmf_params(default_params: dict) -> dict:
    """
    Defines all necessary NMF parameters via streamlit widgets.

    Parameters
    ----------
    default_params : dict
        Contains the default parameters that are used if the created streamlit checkbox is True.

    Returns
    -------
    nmf_params : dict
        A dictionary containing the selected NMF parameters.
    """
    st.sidebar.markdown('# NMF settings')

    # default parameter selection
    use_default = dict(
        n_atoms=st.sidebar.checkbox('Use ground truth number of atoms', True),
        atom_shape=st.sidebar.checkbox('Use ground truth atom shape', True),
    )

    # user-defined parameter values
    n_dims = len(default_params['atom_shape'])
    selected_params = dict(
        n_iterations=st.sidebar.number_input('# Iterations', value=100, min_value=1),
        n_atoms=st.sidebar.number_input(
            '# Atoms', value=default_params['n_atoms'], min_value=1) if not use_default['n_atoms'] else None,
        atom_shape=tuple([st.sidebar.number_input('Atom size', value=default_params['atom_shape'][0], min_value=1)]
                         * n_dims) if not use_default['atom_shape'] else None,
        sparsity_H=st.sidebar.number_input('Activation sparsity', min_value=0.0, value=0.0, step=0.01),
        inhibition_strength=st.sidebar.number_input('Lateral activation inhibition', min_value=0.0, value=0.1, step=0.01),
        backend=st.sidebar.selectbox('Backend', ['numpy', 'numpy_fft', 'numpy_caching_fft', 'pytorch', 'pytorch_fft'], 4),
        reconstruction_mode=st.sidebar.selectbox('Reconstruction', ['valid', 'full', 'circular'], 2)
    )

    # override with selected parameters with defaults
    selected_params.update({k: v for k, v in default_params.items() if use_default[k]})
    return selected_params


class SignalTool(ABC):
    """An abstract base class that serves as a factory for creating specialized objects that facilitate the handling of
    different signal types."""

    def __new__(cls, n_dims: int):
        """
        Parameters
        ----------
        n_dims : int
            The dimensionality of the signals to be managed.
            * n_dims=1 for time series.
            * n_dims=2 for image data.
        """
        if n_dims == 1:
            return super(SignalTool, cls).__new__(SignalTool1D)
        else:
            return super(SignalTool, cls).__new__(SignalTool2D)

    @classmethod
    def st_generate_input(cls) -> Tuple[np.ndarray, dict]:
        """
        Defines all signal parameters via streamlit widgets and returns a generated input matrix V for the NMF together
        with a dictionary containing details of the used NMF atoms.

        Returns
        -------
        V : np.ndarray
            The generated input for the NMF.
        nmf_params : dict
            Ground truth NMF atom parameters that were used for the signal generation.
        """
        st.sidebar.markdown('# Signal settings')
        n_signals = st.sidebar.number_input('# Signals', min_value=1, value=10)
        signal_params = cls.st_define_signal_params()
        V = []
        for _ in range(n_signals):
            signal, W = cls.generate_signal(signal_params)
            V.append(signal)
        V = np.stack(V)
        nmf_params = {'n_atoms': W.shape[0], 'atom_shape': W.shape[2:]}
        return V, nmf_params

    @classmethod
    def st_compare_signals(cls, V: np.ndarray, R: np.ndarray):
        """
        Compares a given input matrix with its NMF reconstruction in streamlit.

        Parameters
        ----------
        V : np.ndarray
            The input that was factorized via NMF.
        R : np.ndarray
            The NMF reconstruction of the input.
        """
        st.markdown('# Global signal reconstruction')

        # show the input, its reconstruction, and the reconstruction error as images next to each other
        cols = st.beta_columns(3)
        for col, X, title in zip(cols, [V, R, V-R], ['Input', 'Reconstruction', 'Error']):
            with col:
                fig = plt.figure()
                plt.imshow(X.reshape(X.shape[0], -1), aspect='auto', interpolation='none')
                plt.title(title)
                st.pyplot(fig)

    @classmethod
    def st_compare_individual_signals(cls, V: np.ndarray, R: np.ndarray):
        """
        Selects a particular signal and its reconstruction from the given input via a streamlit widget and compares them.

        Parameters
        ----------
        V : np.ndarray
            The input that was factorized via NMF.
        R : np.ndarray
            The NMF reconstruction of the input.
        """
        st.markdown('# Individual signal reconstruction')
        i_signal = st.slider('Signal number', 1, V.shape[0]) - 1
        cls._st_compare_individual_signals(V[i_signal], R[i_signal])

    @classmethod
    def st_plot_partial_reconstructions(cls, V: np.ndarray, nmf: TransformInvariantNMF):
        """
        Visualizes the partial reconstructions of the given input by the different NMF atoms.

        Parameters
        ----------
        V : np.ndarray
            The input that was factorized via NMF.
        nmf : TransformInvariantNMF
            The trained NMF object.
        """
        st.markdown('# Partial signal reconstructions')
        i_signal = st.slider('Signal number', 1, V.shape[0], key='i_signal_partial') - 1
        for i_atom in range(nmf.n_atoms):
            R_atom = nmf.R_partial(i_atom)
            cols = st.beta_columns(2)
            for col, signals, signal_opts, opts in zip(
                    cols,
                    [[nmf.W[i_atom]], [R_atom[i_signal], V[i_signal]]],
                    [[{}], [{'label': 'Atom contribution', 'color': 'tab:red', 'linestyle': '--'}, {'label': 'Signal'}]],
                    [{'title': 'Atom'}, {'title': 'Atom contribution'}],
            ):
                with col:
                    cls.plot_signals(signals, signal_opts, opts)

    @classmethod
    @abstractmethod
    def plot_signals(cls, signals: List[np.ndarray], signal_opts: Iterable[dict] = repeat({}), opts: dict = {}):
        """
        Visualizes a given list of signals.

        Parameters
        ----------
        signals : List[np.ndarray]
            The list of signals to be plotted.
        signal_opts : Iterable[dict]
            A list of dictionaries containing plotting options for each individual signal.
        opts : dict
            A dictionary containing global plotting options.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def st_define_signal_params(cls) -> dict:
        """Defines all signal parameters via streamlit widgets and returns them in a dictionary."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def generate_signal(cls, signal_params: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Creates a single signal using the specified signal parameters. Returns the signal and the used NMF atoms."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _st_compare_individual_signals(cls, V_i: np.ndarray, R_i: np.ndarray):
        """
        Compares a single signal and its reconstruction in streamlit.

        Parameters
        ----------
        V_i : np.ndarray
            A single input signal from the input that was factorized via NMF.
        R_i : np.ndarray
            The NMF reconstruction of the given input signal.
        """
        raise NotImplementedError


class SignalTool1D(SignalTool):

    @classmethod
    def st_define_signal_params(cls) -> dict:

        # define the number of channels and pulses
        n_channels = st.sidebar.number_input('# Channels', min_value=1, value=3, max_value=5)
        n_pulses = st.sidebar.number_input('# Pulses', min_value=1, value=3)

        # to avoid having too many different symbols, consider only those where all channels are identical
        shapes = ['n', '-', '^', 'v', '_']
        symbols = [s * n_channels for s in shapes]

        # create the parameter dictionary
        signal_params = dict(
            n_pulses=n_pulses,
            symbols=symbols,
            pulse_length=st.sidebar.number_input('Pulse length', min_value=1, value=20),
        )

        return signal_params

    @classmethod
    def generate_signal(cls, signal_params: dict) -> Tuple[np.ndarray, np.ndarray]:
        signal, W = generate_pulse_train(**signal_params)
        return signal, W

    @classmethod
    def _st_compare_individual_signals(cls, V_i: np.ndarray, R_i: np.ndarray):
        signals = [V_i, R_i]
        opts = [{'label': 'Reconstruction', 'color': 'tab:red', 'linestyle': '--'}, {'label': 'Signal', 'zorder': -1}]
        cls.plot_signals(signals, opts)

    @classmethod
    def plot_signals(cls, signals: List[np.ndarray], signal_opts: Iterable[dict] = repeat({}), opts: dict = {}):
        assert len(np.unique([signal.shape[0] for signal in signals])) == 1
        n_channels = signals[0].shape[0]
        with plt.style.context('seaborn'):
            fig, axs = plt.subplots(nrows=n_channels)
            axs = np.atleast_1d(axs)
            for signal, signal_opt in zip(signals, signal_opts):
                for i_channel, ax in enumerate(axs):
                    ax.plot(signal[i_channel], **signal_opt)
            plt.legend()
            fig.suptitle(opts.get('title'))
            st.pyplot(fig)


class SignalTool2D(SignalTool):

    @classmethod
    def st_define_signal_params(cls) -> dict:

        # choose between grayscale or color images, select the number of symbols per dimension and their size
        n_channels = 1 if st.sidebar.radio('Image type', ['Grayscale', 'Color'], 1) == 'Grayscale' else 3
        n_symbols = st.sidebar.number_input('# Patches', min_value=1, value=5)
        symbol_size = st.sidebar.number_input('Patch size', min_value=1, value=10)

        # create all possible combinations of shapes and color
        shapes = ['+', 'x', 's']
        colors = ['r', 'g', 'b', 'y', 'm', 'c', 'w'] if n_channels == 3 else ['']
        symbols = [''.join(spec) for spec in product(shapes, colors)]

        # create the parameter dictionary
        signal_params = dict(
            n_symbols=n_symbols,
            symbol_size=symbol_size,
            symbols=st.sidebar.multiselect('Symbols', symbols, symbols),
        )

        return signal_params

    @classmethod
    def generate_signal(cls, signal_params: dict) -> Tuple[np.ndarray, np.ndarray]:
        signal, W = generate_block_image(**signal_params)
        return signal, W

    @classmethod
    def _st_compare_individual_signals(cls, V_i: np.ndarray, R_i: np.ndarray):
        cols = st.beta_columns(2)
        for col, X, title in zip(cols, [V_i, R_i], ['Input', 'Reconstruction']):
            with col:
                cls.plot_signals([X], opts={'title': title})

    @classmethod
    def plot_signals(cls, signals: List[np.ndarray], signal_opts: Iterable[dict] = repeat({}), opts: dict = {}):
        fig = plt.figure()
        plt.imshow(signals[0].transpose((1, 2, 0)) / signals[0].max())
        plt.title(opts.get('title'))
        st.pyplot(fig)
