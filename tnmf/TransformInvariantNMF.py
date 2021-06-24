"""
Transform-Invariant Non-Negative Matrix Factorization

Authors: Adrian Šošić, Mathias Winkel
"""

# TODO: backend-specific return types
# TODO: no atom_shape for non-ShiftInvariantNMF
# TODO: handling of transform input/output shapes
# TODO: naming convention: energy (instead of cost/error)
# TODO: add options for tensor renormalization
# TODO: cache reconstruction result
# TODO: flexible input types for V
# TODO: we extract .shape[...] too often
# TODO: add support for inhibition

import logging
from typing import Tuple, Callable, Union

import numpy as np

from .backends.NumPy import NumPy_Backend
from .backends.NumPy_FFT import NumPy_FFT_Backend
from .backends.PyTorch import PyTorch_Backend
from .backends.PyTorch_FFT import PyTorch_FFT_Backend
from .backends.NumPy_CachingFFT import NumPy_CachingFFT_Backend


class TransformInvariantNMF:
    r"""
    Transform Invariant Non-Negative Matrix Factorization.

    Finds non-negative tensors :attr:`W` (dictionary) and :attr:`H` (activations) that approximate the non-negative tensor
    :attr:`V` (samples) for a given transform operator`. # TODO: add link to TNMF model

    .. note::
        Currently, only a single transform type, corresponding to *shift invariance*, is supported and hard-coded. In contrast
        to other *generic* types of transforms, shift invariance can be efficiently achieved through convolution operations
        (or, equivalently, multiplication in Fourier domain). Therefore, shift invariance will remain hard-coded and retained
        as an optional transform type even when additional transforms become supported in future releases.

    Optimization is performed via multiplicative updates to :attr:`W` and :attr:`H`, see [1]_.
    Different backend implementations (NumPy, PyTorch, with/without FFT, etc.) can be selected by the user.

    Parameters
    ----------
    n_atoms : int
        Number of elementary atoms. The shape of :attr:`W` will be ``(n_atoms, n_channels, *atom_shape)``.
    atom_shape : Tuple[int, ...]
        Shape of the elementary atoms. The shape of :attr:`W` will be ``(n_atoms, n_channels, *atom_shape)``.
    inhibition_range: Union[int, Tuple[int, ...]], default = None
        Lateral inhibition range. If set to None, the value is set to ``*atom_shape``, which ensures
        that activations are pairwise sufficiently far apart, that the corresponding atoms do not overlap
        in the reconstruction.
    n_iterations : int, default = 1000
        Maximum number of iterations (:attr:`W` and :attr:`H` updates) to be performed.
    backend : {'numpy', 'numpy_fft', 'numpy_caching_fft', 'pytorch'}, default = 'numpy_fft'
        Defines the optimization backend.

        * `'numpy'`: Selects the :class:`.NumPy_Backend`.
        * `'numpy_fft'` (default): Selects the :class:`.NumPy_FFT_Backend`.
        * `'numpy_caching_fft'`: Selects the :class:`.NumPy_CachingFFT_Backend`.
        * `'pytorch'`: Selects the :class:`.PyTorch_Backend`.
        * `'pytorch_fft'`: Selects the :class:`.PyTorch_FFT_Backend`.

    logger : logging.Logger, default = None
        Logger instance used for intermediate output. If None, an internal logger instance will be created.
    verbose : {0, 1, 2, 3}, default = 0
        Verbosity level.

        * 0: Show only errors.
        * 1: Include warnings.
        * 2: Include info.
        * 3: Include debug messages.

    **kwargs
        Keyword arguments that are handed to the constructor of the backend class.

    Attributes
    ----------
    W : np.ndarray
        The dictionary tensor of shape ``(n_atoms, num_channels, *atom_shape)``.
    H : np.ndarray
        The activation tensor of shape ``(num_samples, num_atoms, *shift_shape)``.
    R : np.ndarray
        The reconstruction of the sample tensor using the current :attr:`W` and :attr:`H`.

    Examples
    --------
        TODO: add examples

    Notes
    -----
    Planned features:

        * Batch processing
        * Arbitrary transform types
        * Nested transformations
        * Additional reconstruction norms

    References
    ----------
    # TODO: add bibtex file

    .. [1] Lee, D.D., Seung, H.S., 2000. Algorithms for Non-negative Matrix Factorization,
        in: Proceedings of the 13th International Conference on Neural Information
        Processing Systems. pp. 535–541. https://doi.org/10.5555/3008751.3008829
    """

    def __init__(
            self,
            n_atoms: int,
            atom_shape: Tuple[int, ...],
            inhibition_range: Union[int, Tuple[int, ...]] = None,
            n_iterations: int = 1000,
            backend: str = 'numpy_fft',
            logger: logging.Logger = None,
            verbose: int = 0,
            **kwargs,
    ):
        self.atom_shape = atom_shape

        if inhibition_range is None:
            # default inhibition range = minimal range to cover the atom size
            self._inhibition_range = tuple(a-1 for a in atom_shape)
        elif isinstance(inhibition_range, int):
            self._inhibition_range = (inhibition_range, ) * len(atom_shape)
        else:
            self._inhibition_range = inhibition_range

        assert len(self._inhibition_range) == len(atom_shape)
        self._inhibition_kernels_1D = tuple((1 - ((np.arange(-i, i + 1) / (i+1)) ** 2) for i in self._inhibition_range))
        self.n_atoms = n_atoms
        self.n_iterations = n_iterations
        self._axes_W_normalization = tuple(range(-len(atom_shape), 0))
        self.eps = 1.e-9

        backend_map = {
            'numpy': NumPy_Backend,
            'numpy_fft': NumPy_FFT_Backend,
            'numpy_caching_fft': NumPy_CachingFFT_Backend,
            'pytorch': PyTorch_Backend,
            'pytorch_fft': PyTorch_FFT_Backend,
        }

        self._backend = backend_map[backend.lower()](**kwargs)

        self._logger = logger if logger is not None else logging.getLogger(self.__class__.__name__)
        self._logger.setLevel([logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG][verbose])
        self._logger.debug(f'Using {backend} backend.')

        self.n_iterations_done = 0
        self._W = None
        self._H = None

    @property
    def W(self) -> np.ndarray:
        return self._backend.to_ndarray(self._W)

    @property
    def H(self) -> np.ndarray:
        return self._backend.to_ndarray(self._H)

    @property
    def R(self) -> np.ndarray:
        return self._backend.to_ndarray(self._reconstruct())

    def R_partial(self, i_atom: int) -> np.ndarray:
        return self._backend.to_ndarray(self._backend.partial_reconstruct(self._W, self._H, i_atom))

    def _reconstruct(self) -> np.ndarray:
        return self._backend.reconstruct(self._W, self._H)

    def _energy_function(self, V: np.ndarray) -> float:
        return self._backend.reconstruction_energy(V, self._W, self._H)

    def _multiplicative_update(self, arr: np.ndarray, neg, pos, sparsity: float = 0.):
        assert sparsity >= 0

        regularization = self.eps

        if sparsity > 0:
            regularization += sparsity

        pos += regularization

        arr *= neg
        arr /= pos

    def _update_W(self, V: np.ndarray):
        neg, pos = self._backend.reconstruction_gradient_W(V, self._W, self._H)
        self._multiplicative_update(self._W, neg, pos)
        self._backend.normalize(self._W, axis=self._axes_W_normalization)

    def _update_H(self, V: np.ndarray, sparsity: float = 0., inhibition: float = 0.):
        # TODO: sparsity and inhibition should be handled by the backends
        neg, pos = self._backend.reconstruction_gradient_H(V, self._W, self._H)

        # add the inhibition gradient component
        if inhibition > 0:
            # TODO: maybe also add cross-channel/cross-atom inhibition?
            convolve_axes = range(-len(self.atom_shape), 0)
            inhibition_gradient = self._backend.convolve_multi_1d(self._H, self._inhibition_kernels_1D, convolve_axes)
            inhibition_gradient -= self._H
            inhibition_gradient *= inhibition
            pos += inhibition_gradient

        self._multiplicative_update(self._H, neg, pos, sparsity)

    def _do_fit(
            self,
            V: np.ndarray,
            update_H: bool,
            update_W: bool,
            sparsity_H: float,
            inhibition_strength: float,
            keep_W: bool,
            progress_callback: Callable[['TransformInvariantNMF', int], bool],
    ):
        assert update_H or update_W
        assert sparsity_H >= 0
        assert inhibition_strength >= 0

        self._W, self._H = self._backend.initialize(
            V, self.atom_shape, self.n_atoms, self._W if keep_W else None)

        if not keep_W:
            self._backend.normalize(self._W, self._axes_W_normalization)

        for self.n_iterations_done in range(self.n_iterations):
            if update_H:
                self._update_H(V, sparsity_H, inhibition_strength)

            if update_W:
                self._update_W(V)

            if progress_callback is not None:
                if not progress_callback(self, self.n_iterations_done):
                    break
            else:
                self._logger.info(f"Iteration: {self.n_iterations_done}\tEnergy function: {self._energy_function(V)}")

        self._logger.info("NMF finished.")

    def fit(
            self,
            V: np.ndarray,
            update_H: bool = True,
            update_W: bool = True,
            sparsity_H: float = 0.,
            inhibition_strength: float = 0.,
            progress_callback: Callable[['TransformInvariantNMF', int], bool] = None,
    ):
        r"""
        Perform non-negative matrix factorization of samples :attr:`V`, i.e. optimization of dictionary :attr:`W` and
        activations :attr:`H`.

        Parameters
        ----------
        V : np.ndarray
            Samples to be reconstructed. The shape of the sample tensor is ``(n_samples, n_channels, *sample_shape)``,
            where `sample_shape` is the shape of the individual samples and each sample consists of `n_channels`
            individual channels.
        update_H : bool, default = True
            If False, the activation tensor :attr:`H` will not be updated.
        update_W : bool, default = True
            If False, the dictionary tensor :attr:`W' will not be updated.
        sparsity_H : float, default = 0.
            Sparsity enforcing regularization for the :attr:`H` update.
        inhibition_strength : float, default = 0.
            Lateral inhibition regularization factor for the :attr:`H` update.
        progress_callback : Callable[['TransformInvariantNMF', int], bool], default = None
            If provided, this function will be called after every iteration, i.e. after every update to :attr:`H` and
            :attr:`W`. The first parameter to the function is the calling :class:`TransformInvariantNMF` instance, which can be
            used to inspect intermediate results, etc. The second parameter is the current iteration step.

            If the `progress_callback` function returns False, iteration will be aborted, which allows to implement
            specialized convergence criteria.
        """
        self._do_fit(V, update_H, update_W, sparsity_H, inhibition_strength, False, progress_callback)

    def partial_fit(
            self,
            V: np.ndarray,
            update_H: bool = True,
            update_W: bool = True,
            sparsity_H: float = 0.,
            inhibition_strength: float = 0.,
            progress_callback: Callable[['TransformInvariantNMF', int], bool] = None,
    ):
        self._do_fit(V, update_H, update_W, sparsity_H, inhibition_strength, self.n_iterations_done > 0, progress_callback)
