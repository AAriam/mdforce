"""
This module contains functions for calculating the distance vectors and distances between different
arrays.
"""

# Standard library
from typing import Tuple

# 3rd-party packages
import numpy as np


def components_single_single(
        q_i_x: float, q_i_y: float, q_i_z: float, q_j_x: float, q_j_y: float, q_j_z: float
) -> Tuple[float, float, float, float]:
    # Calculate distance vector components
    q_ji_x = q_i_x - q_j_x
    q_ji_y = q_i_y - q_j_y
    q_ji_z = q_i_z - q_j_z
    # Calculate the norm of vector (i.e. distance)
    d_ij = (q_ji_x ** 2 + q_ji_y ** 2 + q_ji_z ** 2) ** 0.5
    return q_ji_x, q_ji_y, q_ji_z, d_ij


def two_arrays(q_i: np.ndarray, q_j: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the distance vector 'q_ji = q_i - q_j' and the distances 'd_ij = ||q_ji||'. Each of
    'q_i' and 'q_j' can either be a single vector (i.e. 1D-array), or an n-dimensional array of
    vectors, but in any case, they should have the same number of elements in their last
    dimensions. The calculated array of distance-vectors will then have the same dimension as the
    higher-dimensional array between the two, and the array of distances will have one dimension
    less. The distances are calculated for the last dimension.

    Parameters
    ----------
    q_i : numpy.ndarray
        The first array.
    q_j : numpy.ndarray
        The second array.

    Returns
    -------
    q_ji, d_ij : Tuple[numpy.ndarray, numpy.ndarray]
        The array of distance vectors 'q_ji', followed by the array of distances 'd_ij'.
        """
    # Calculate distance vectors
    q_ji = q_i - q_j
    # Calculate the norm of vectors (i.e. distances)
    d_ij = np.linalg.norm(q_ji, axis=q_ji.ndim - 1)
    return q_ji, d_ij


def array_multi_self(q: np.ndarray):
    # Create an empty matrix for storing the distance vectors
    num_vects, num_dims = q.shape
    q_jsis = np.empty((num_vects, num_vects, num_dims))
    d_isjs = np.empty((num_vects, num_vects))
    # Iterate over all vectors, except the last one
    for i, q_i in enumerate(q[:-1]):
        # Calculate the distance vectors between current vector and all vectors after it
        dist_vectors = q_i - q[i+1:]
        q_jsis[i, i+1:] = dist_vectors
        d_isjs[i, i+1:] = np.linalg.norm(dist_vectors, axis=1)
    return q_jsis, d_isjs


def array_multi_self_pbc(q: np.ndarray, indices_first_long_range, pbc_box_lengths):
    # Create an empty matrix for storing the distance vectors
    num_vects, num_dims = q.shape
    q_jsis = np.empty((num_vects, num_vects, num_dims))
    d_isjs = np.empty((num_vects, num_vects))
    # Iterate over all vectors, except the last one
    for (i, q_i), idx_first_long_range in zip(enumerate(q[:-1]), indices_first_long_range):
        # Calculate the distance vectors between current vector and all vectors after it
        dist_vectors = q_i - q[i+1:]
        dist_vectors[idx_first_long_range-i-1:] -= pbc_box_lengths * np.rint(
            dist_vectors[idx_first_long_range - i - 1:] / pbc_box_lengths
        )
        q_jsis[i, i+1:] = dist_vectors
        d_isjs[i, i+1:] = np.linalg.norm(dist_vectors, axis=1)
    return q_jsis, d_isjs
