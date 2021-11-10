"""
Transform-Invariant Non-Negative Matrix Factorization

Authors: Adrian Šošić, Mathias Winkel
"""

# TODO: backend-specific return types
# TODO: no atom_shape for non-ShiftInvariantNMF
# TODO: handling of transform input/output shapes
# TODO: add options for tensor renormalization
# TODO: cache reconstruction result
# TODO: flexible input types for V

import logging
from itertools import islice, count
from typing import Iterable, Tuple, Callable, Union, Iterator
from enum import Enum

import numpy as np

from .backends.NumPy import NumPy_Backend
from .backends.NumPy_FFT import NumPy_FFT_Backend
from .backends.PyTorch import PyTorch_Backend
from .backends.PyTorch_FFT import PyTorch_FFT_Backend
from .backends.NumPy_CachingFFT import NumPy_CachingFFT_Backend


def _compute_sequential_minibatches(length: int, batch_size: int) -> Iterable[slice]:
    if batch_size is None:
        yield slice(None)
    else:
        start = 0
        while start < length:
            end = min(length, start + batch_size)
            yield slice(start, end)
            start = end


def _random_shuffle(arr):
    np.random.shuffle(np.asarray(arr))
    return arr


class MiniBatchAlgorithm(Enum):
    r"""
    MiniBatch algorithms that can be used with :meth:`.TransformInvariantNMF.fit_minibatch`.
    """
    Basic_MU = 3   # Algorithm 3 Basic alternating scheme for MU rules
    Cyclic_MU = 4  # Algorithm 4 Cyclic mini-batch for MU rules
    ASG_MU = 5     # Algorithm 5 Asymmetric SG mini-batch MU rules (ASG-MU)
    GSG_MU = 6     # Algorithm 6 Greedy SG mini-batch MU rules (GSG-MU)
    ASAG_MU = 7    # Algorithm 7 Asymmetric SAG mini-batch MU rules (ASAG-MU)
    GSAG_MU = 8    # Algorithm 8 Greedy SAG mini-batch MU rules (GSAG-MU)


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
    Minibatch updates are possible via a selection of algorithms from [2]_.
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

    .. [1] D.D. Lee, H.S. Seung, 2000. Algorithms for Non-negative Matrix Factorization,
        in: Proceedings of the 13th International Conference on Neural Information
        Processing Systems. pp. 535–541. https://doi.org/10.5555/3008751.3008829
    .. [2] R. Serizel, S. Essid, G. Richard, 2016. Mini-batch stochastic approaches for
        accelerated multiplicative updates in nonnegative matrix factorisation with
        beta-divergence, in: 26th International Workshop on Machine Learning for Signal
        Processing (MLSP). pp 1-6. http://ieeexplore.ieee.org/document/7738818/
    """

    def __init__(
            self,
            n_atoms: int,
            atom_shape: Tuple[int, ...],
            inhibition_range: Union[int, Tuple[int, ...]] = None,
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

        self._W = None
        self._H = None
        self._V = None  # V is not necessarily identical with the V outside as the stochastic minibatch methods shuffle V

    @property
    def W(self) -> np.ndarray:
        return self._backend.to_ndarray(self._W)

    @property
    def H(self) -> np.ndarray:
        return self._backend.to_ndarray(self._H)

    @property
    def V(self) -> np.ndarray:
        return self._V

    @property
    def R(self) -> np.ndarray:
        return self._backend.to_ndarray(self._reconstruct())

    def R_partial(self, i_atom: int) -> np.ndarray:
        return self._backend.to_ndarray(self._backend.partial_reconstruct(self._W, self._H, i_atom))

    def _reconstruct(self) -> np.ndarray:
        return self._backend.reconstruct(self._W, self._H)

    def _energy_function(self) -> float:
        return self._backend.reconstruction_energy(self._V, self._W, self._H)

    def _multiplicative_update(self, arr: np.ndarray, neg, pos, sparsity: float = 0.):
        assert sparsity >= 0

        regularization = self.eps

        if sparsity > 0:
            regularization += sparsity

        pos += regularization

        arr *= neg
        arr /= pos

    def _update_W(self, s: slice):
        neg, pos = self._backend.reconstruction_gradient_W(self._V, self._W, self._H, s)
        assert neg.shape == self._W.shape
        assert pos.shape == self._W.shape
        self._multiplicative_update(self._W, neg, pos)
        self._backend.normalize(self._W, axis=self._axes_W_normalization)

    def _update_H(self, s: slice, sparsity: float = 0., inhibition: float = 0., cross_inhibition: float = 0):
        # TODO: sparsity and inhibition computation should be handled by the backends
        neg, pos = self._backend.reconstruction_gradient_H(self._V, self._W, self._H, s)
        assert neg.shape == self._H[s].shape
        assert pos.shape == self._H[s].shape

        # add the inhibition gradient component
        if inhibition > 0 or cross_inhibition > 0:
            convolve_axes = range(-len(self.atom_shape), 0)
            inhibition_gradient = self._backend.convolve_multi_1d(self._H[s], self._inhibition_kernels_1D, convolve_axes)

            if inhibition > 0:
                tmp = inhibition_gradient - self.H[s]  # prevent the atom from suppressing itself at its own position
                tmp *= inhibition
                pos += tmp
            if cross_inhibition > 0:
                # sum inhibition gradient over all atoms
                tmp = inhibition_gradient.sum(axis=1, keepdims=True)
                # broadcast the summed inhibition gradient to all atoms and subtract the contribution of the current atom
                # thus, tmp contains for every atom the sum over all the inhibition gradient of all other atoms now
                tmp = -inhibition_gradient + tmp
                # we scale with (number_atoms - 1)
                tmp *= cross_inhibition / (self.n_atoms - 1)
                pos += tmp

        self._multiplicative_update(self._H[s], neg, pos, sparsity)

    def _initialize_matrices(self, V: np.ndarray, keep_W: bool):
        self._V = V
        self._W, self._H = self._backend.initialize(
            self._V, self.atom_shape, self.n_atoms, self._W if keep_W else None,
            self._axes_W_normalization)

    def fit_basic(
            self,
            V: np.ndarray,
            n_iterations: int = 1000,
            update_H: bool = True,
            update_W: bool = True,
            keep_W: bool = False,
            sparsity_H: float = 0.,
            inhibition_strength: float = 0.,
            cross_atom_inhibition_strength: float = 0.,
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
        n_iterations : int, default = 1000
            Maximum number of iterations (:attr:`W` and :attr:`H` updates) to be performed.
        update_H : bool, default = True
            If False, the activation tensor :attr:`H` will not be updated.
        update_W : bool, default = True
            If False, the dictionary tensor :attr:`W` will not be updated.
        keep_W : bool, default = False
            If False, the dictionary tensor :attr:`W` will not be (re)initialized before starting iteration.
        sparsity_H : float, default = 0.
            Sparsity enforcing regularization for the :attr:`H` update.
        inhibition_strength : float, default = 0.
            Lateral inhibition regularization factor for the :attr:`H` update within the same atom.
        cross_atom_inhibition_strength : float, default = 0.
            Lateral inhibition regularization factor for the :attr:`H` update across different atoms.
        progress_callback : Callable[['TransformInvariantNMF', int], bool], default = None
            If provided, this function will be called after every iteration, i.e. after every update to :attr:`H` and
            :attr:`W`. The first parameter to the function is the calling :class:`TransformInvariantNMF` instance, which can be
            used to inspect intermediate results, etc. The second parameter is the current iteration step.

            If the `progress_callback` function returns False, iteration will be aborted, which allows to implement
            specialized convergence criteria.
        """
        assert np.all(V >= 0)
        assert update_H or update_W
        assert sparsity_H >= 0
        assert inhibition_strength >= 0
        assert cross_atom_inhibition_strength >= 0
        keep_W = False

        self._initialize_matrices(V, keep_W)

        for iteration in range(n_iterations):
            if update_H:
                self._update_H(slice(None), sparsity_H, inhibition_strength, cross_atom_inhibition_strength)

            if update_W:
                self._update_W(slice(None))

            if progress_callback is not None:
                if not progress_callback(self, iteration):
                    break
            else:
                self._logger.info(f"Iteration: {iteration}\tEnergy function: {self._energy_function()}")

        self._logger.info("TNMF finished.")

    def fit_minibatch(
            self,
            V: np.ndarray,
            algorithm: MiniBatchAlgorithm = MiniBatchAlgorithm.Basic_MU,
            batch_size: int = 3,
            max_epoch: int = 1000,   # corresponds to max_iter if algorithm == MiniBatchAlgorithm.Basic_MU
            sag_lambda: float = 0.8,
            keep_W: bool = False,
            sparsity_H: float = 0.,
            inhibition_strength: float = 0.,
            cross_atom_inhibition_strength: float = 0.,
            progress_callback: Callable[['TransformInvariantNMF', int], bool] = None,
    ):
        r"""
        Perform non-negative matrix factorization of samples :attr:`V`, i.e. optimization of dictionary :attr:`W` and
        activations :attr:`H` via mini-batch updates using a selection of algorithms from [3]_.

        Parameters
        ----------
        V : np.ndarray
            Samples to be reconstructed. The shape of the sample tensor is ``(n_samples, n_channels, *sample_shape)``,
            where `sample_shape` is the shape of the individual samples and each sample consists of `n_channels`
            individual channels.
        algorithm: MiniBatchAlgorithm
            MiniBatch update scheme to be used. See :class:`MiniBatchAlgorithm` and [3]_ for the different choices.
        batch_size: int, default = 3
            Number of samples per mini batch. Ignored if algorithm==MiniBatchAlgorithm.Basic_MU
        max_epoch: int, default = 1000
            Maximum number of epochs (iterations if algorithm==MiniBatchAlgorithm.Basic_MU) across the full
            sample set to be performed.
        sag_lambda: float, default = 0.8
            Exponential forgetting factor for for the stochastic _average_ gradient updates, i.e.
            MiniBatchAlgorithm.ASAG_MU and MiniBatchAlgorithm.GSAG_MU
        keep_W : bool, default = False
            If False, the dictionary tensor :attr:`W` will not be (re)initialized before starting iteration.
        sparsity_H : float, default = 0.
            Sparsity enforcing regularization for the :attr:`H` update.
        inhibition_strength : float, default = 0.
            Lateral inhibition regularization factor for the :attr:`H` update within the same atom.
        cross_atom_inhibition_strength : float, default = 0.
            Lateral inhibition regularization factor for the :attr:`H` update across different atoms.
        progress_callback : Callable[['TransformInvariantNMF', int], bool], default = None
            If provided, this function will be called after every (epoch-)iteration.
            The first parameter to the function is the calling :class:`TransformInvariantNMF` instance, which can be
            used to inspect intermediate results, etc. The second parameter is the current iteration step.

            If the `progress_callback` function returns False, (epoch-)iteration will be aborted, which allows to implement
            specialized convergence criteria.
        References
        ----------
        .. [3] R. Serizel, S. Essid, G. Richard, 2016. Mini-batch stochastic approaches for
            accelerated multiplicative updates in nonnegative matrix factorisation with
            beta-divergence, in: 26th International Workshop on Machine Learning for Signal
            Processing (MLSP). pp 1-6. http://ieeexplore.ieee.org/document/7738818/
        """
        assert np.all(V >= 0)
        assert sparsity_H >= 0
        assert inhibition_strength >= 0
        assert cross_atom_inhibition_strength >= 0
        assert isinstance(algorithm, MiniBatchAlgorithm)

        stochastic_update = algorithm in (5, 6, 7, 8)
        self._initialize_matrices(V if not stochastic_update else _random_shuffle(V.copy()), keep_W)

        batches = list(_compute_sequential_minibatches(len(self._V), batch_size))

        epoch_update = {
            MiniBatchAlgorithm.Basic_MU: self._epoch_update_algorithm_3,
            MiniBatchAlgorithm.Cyclic_MU: self._epoch_update_algorithm_4,
            MiniBatchAlgorithm.ASG_MU: self._epoch_update_algorithm_5,
            MiniBatchAlgorithm.GSG_MU: self._epoch_update_algorithm_6,
            MiniBatchAlgorithm.ASAG_MU: self._epoch_update_algorithm_7,
            MiniBatchAlgorithm.GSAG_MU: self._epoch_update_algorithm_8,
        }

        kwargs_update_H = dict(
            sparsity=sparsity_H,
            inhibition=inhibition_strength,
            cross_inhibition=cross_atom_inhibition_strength,
        )

        inner_stat = None
        for epoch in range(max_epoch):
            inner_stat = epoch_update[algorithm](
                inner_stat, batches,
                kwargs_update_H,
                sag_lambda)

            if progress_callback is not None:
                if not progress_callback(self, epoch):
                    break
            else:
                self._logger.info(f"{'Iteration' if algorithm == MiniBatchAlgorithm.Basic_MU else 'Epoch'}: {epoch}\t"
                                  f"Energy function: {self._energy_function()}")

        self._logger.info("MiniBatch TNMF finished.")

    def _accumulate_gradient_W(self, gradW_neg, gradW_pos, sag_lambda: float, s: slice):
        neg, pos = self._backend.reconstruction_gradient_W(self._V, self._W, self._H, s)
        if sag_lambda == 1:
            gradW_neg += neg
            gradW_pos += pos
        else:
            gradW_neg *= (1-sag_lambda)
            gradW_pos *= (1-sag_lambda)
            gradW_neg += sag_lambda * neg
            gradW_pos += sag_lambda * pos

        return gradW_neg, gradW_pos

    def _epoch_update_algorithm_3(self, _, ___, args_update_H, __):
        # update H for all samples
        self._update_H(slice(None), **args_update_H)
        # update W after processing all batches using all samples
        self._update_W(slice(None))

    def _epoch_update_algorithm_4(self, _, batches, args_update_H, __):
        gradW_neg, gradW_pos = 0, 0
        for batch in batches:
            # update H for all batches
            self._update_H(batch, **args_update_H)
            # accumulate the gradient over all batches
            gradW_neg, gradW_pos = self._accumulate_gradient_W(gradW_neg, gradW_pos, 1., batch)
        # update W with the gradient that has been accumulated over all batches
        self._multiplicative_update(self._W, gradW_neg, gradW_pos)
        self._backend.normalize(self._W, axis=self._axes_W_normalization)

    def _epoch_update_algorithm_5(self, _, batches, args_update_H, __):
        for batch in _random_shuffle(batches):
            # update H for every batch
            self._update_H(batch, **args_update_H)
            # update W after every batch
            self._update_W(batch)

    def _epoch_update_algorithm_6(self, _, batches, args_update_H, __):
        batch = slice(0, 0)  # initialize to an empty slice as we use it after the loop (just in case batches is empty)
        for batch in _random_shuffle(batches):
            # update H for every batch
            self._update_H(batch, **args_update_H)
        # update W after processing all batches using the last batch (algorithm 6)
        self._update_W(batch)

    def _epoch_update_algorithm_7(self, inner_stat, batches, args_update_H, sag_lambda):
        if inner_stat is None:
            inner_stat = (0, 0)
        for batch in _random_shuffle(batches):
            # update H for every batch
            self._update_H(batch, **args_update_H)
            # average the gradient over all batches and epochs
            inner_stat = self._accumulate_gradient_W(*inner_stat, sag_lambda, batch)
            # update W with the gradient that has been averaged over all batches until now
            self._multiplicative_update(self._W, *inner_stat)
            self._backend.normalize(self._W, axis=self._axes_W_normalization)
        return inner_stat

    def _epoch_update_algorithm_8(self, inner_stat, batches, args_update_H, sag_lambda):
        if inner_stat is None:
            inner_stat = (0, 0)
        batch = slice(0, 0)  # initialize to an empty slice as we use it after the loop (just in case batches is empty)
        for batch in _random_shuffle(batches):
            # update H for every batch
            self._update_H(batch, **args_update_H)
        # average the gradient from the last batch over all epochs
        inner_stat = self._accumulate_gradient_W(*inner_stat, sag_lambda, batch)
        # update W with the gradient that has been averaged over all batches until now
        self._multiplicative_update(self._W, *inner_stat)
        self._backend.normalize(self._W, axis=self._axes_W_normalization)
        return inner_stat

    def fit_subsamples(
            self,
            V: Iterator[np.ndarray],
            subsample_size: int = 3,
            max_subsamples: int = None,
            **kwargs
    ):
        for isub in count(0):
            subsample = list(islice(V, subsample_size))
            if len(subsample) > 0:
                self._logger.info(f"Processing subsample {isub}.")
                self.fit(np.asarray(subsample), keep_W=True, **kwargs)
                if max_subsamples is not None and isub == max_subsamples - 1:
                    self._logger.info("Processed {max_subsamples} subsamples. TNMF on iterator will stop.")
                    return
            else:
                self._logger.info("Sample iterator exhausted. TNMF on full iterator finished.")
                return

    def fit(self, V: np.ndarray, **kwargs):
        if 'subsample_size' in kwargs or 'max_subsamples' in kwargs:
            self.fit_subsamples(iter(V), **kwargs)
        elif 'batch_size' in kwargs or 'algorithm' in kwargs:
            self.fit_minibatch(V, **kwargs)
        else:
            self.fit_basic(V, **kwargs)
