from abc import ABC, abstractmethod
from itertools import product, repeat

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from tnmf.utils.signals import generate_pulse_train


def st_define_nmf_params(n_dims: int) -> dict:
    """
    Defines all necessary NMF parameters via streamlit widgets.

    Parameters
    ----------
    n_dims : int
        The number of dimensions of the input signals.

    Returns
    -------
    nmf_params : dict
        A dictionary containing the selected NMF parameters.
    """
    st.sidebar.markdown('# NMF settings')
    nmf_params = dict(
        n_iterations=st.sidebar.number_input('# Iterations', value=100, min_value=1),
        n_atoms=st.sidebar.number_input('# Atoms', value=3, min_value=1),
        atom_shape=tuple([st.sidebar.number_input('Atom size', value=10, min_value=1)] * n_dims),
    )
    return nmf_params


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
            raise ValueError

    @classmethod
    def st_generate_input(cls) -> np.ndarray:
        """Defines all signal parameters via streamlit widgets and returns a generated input matrix V for the NMF."""
        st.sidebar.markdown('# Signal settings')
        n_signals = st.sidebar.number_input('# Signals', min_value=1, value=10)
        signal_params = cls.st_define_signal_params()
        V = np.stack([cls.generate_signal(signal_params) for _ in range(n_signals)])
        return V

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
        mode = st.radio('Visualization mode', ['Compare', 'Difference'])

        # show the input and its reconstruction as images next to each other
        if mode == 'Compare':
            cols = st.beta_columns(2)
            for col, X in zip(cols, [V, R]):
                with col:
                    fig = plt.figure()
                    plt.imshow(X.reshape(X.shape[0], -1), aspect='auto', interpolation='none')
                    st.pyplot(fig)

        # show the difference between the input and its reconstruction as an image
        else:
            fig = plt.figure()
            plt.imshow((V - R).reshape(V.shape[0], -1), cmap='RdBu', aspect='auto', interpolation='none')
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
        i_signal = st.slider('Signal number', 0, V.shape[0] - 1)
        cls._st_compare_individual_signals(V[i_signal], R[i_signal])

    @classmethod
    @abstractmethod
    def st_define_signal_params(cls) -> dict:
        """Defines all signal parameters via streamlit widgets and returns them in a dictionary."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def generate_signal(cls, signal_params: dict) -> np.ndarray:
        """Creates a single signal using the specified signal parameters."""
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
        symbols = [''.join(chars) for chars in product(*repeat(shapes, n_channels))]

        # create the parameter dictionary
        signal_params = dict(
            n_pulses=n_pulses,
            symbols=symbols,
            pulse_length=st.sidebar.number_input('Pulse length', min_value=1, value=20),
        )

        return signal_params

    @classmethod
    def generate_signal(cls, signal_params: dict) -> np.ndarray:
        signal, _ = generate_pulse_train(**signal_params)
        return signal

    @classmethod
    def _st_compare_individual_signals(cls, V_i: np.ndarray, R_i: np.ndarray):
        n_channels = V_i.shape[0]
        fig, axs = plt.subplots(nrows=n_channels)
        axs = np.atleast_1d(axs)
        for i_channel, ax in enumerate(axs):
            ax.plot(V_i[i_channel], label='signal')
            ax.plot(R_i[i_channel], '--', label='reconstruction', color='tab:red')
        plt.legend()
        st.pyplot(fig)
