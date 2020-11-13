"""
Author: Adrian Sosic
"""

import numpy as np


def normalize(arr: np.array, axis: int = 0) -> np.array:
	"""
	Normalizes a given array such that the sum of its values along the specified axis sum to one.

	Parameters
	----------
	arr : np.array
		The array to be normalized.
	axis : int
		The axis along which the array shall be normalized.

	Returns
	-------
	The normalized array.
	"""
	return arr / np.sum(arr, axis=axis, keepdims=True)
