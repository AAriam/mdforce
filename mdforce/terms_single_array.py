"""
Implementation of the individual terms of a general force field.

Each function calculates the force on a single target particle `i`, due to another single particle
`j` (except for `angle_vibration_harmonic`, which takes in three particles and calculates the force
on each of them). Moreover, each function also returns the potential energy of the system of
particles.

These functions are mostly intended for testing purposes; They are one step more advanced than the
equivalent functions in module `terms_single_simple`, in the sense that they accept the position
vector arguments as numpy arrays.
"""


# Standard library
from typing import Tuple

# 3rd-party
import numpy as np

# Self
from . import distances


def coulomb(
    q_i: np.ndarray, q_j: np.ndarray, c_i: float, c_j: float, k_e: float
) -> Tuple[np.ndarray, float]:
    """
    Calculate the coulomb potential between two particles 'i' and 'j', and force on 'i' due to 'j'.

    Parameters
    ----------
    q_i : numpy.ndarray
        Coordinates vector of particle 'i' as a 1D-array of shape (n, ), where 'n' is the number of
        spatial dimensions. The calculated force will be for this particle.
    q_j : numpy.ndarray
        Coordinates vector of particle 'j' as a 1D-array of shape (n, ), where 'n' is the number of
        spatial dimensions.
    c_i : float
        Charge of particle 'i'.
    c_j : float
        Charge of particle 'j'.
    k_e : float
        Coulomb constant, i.e. (1 / 4πε0).

    Returns
    -------
    force_i, potential_ij : Tuple[numpy.ndarray, float]
        Force vector for particle 'i' due to particle 'j', followed by the potential energy between
        the two particles.

    Notes
    -----
    The force vector for particle 'j' due to particle 'i' is the same vector as the return value,
    only with opposite signs for all components, whereas the potential energy does not change.
    """
    # Calculate distance vector and its norm (i.e. distance)
    q_ji, d_ij = distances.two_arrays(q_i, q_j)
    # Calculate potential
    e_ij = k_e * c_i * c_j / d_ij
    # Calculate force
    f_i = k_e * c_i * c_j / d_ij ** 3 * q_ji
    return f_i, e_ij


def lennard_jones(
    q_i: np.ndarray, q_j: np.ndarray, a_ij: float, b_ij: float
) -> Tuple[np.ndarray, float]:
    """
    Calculate the Lennard-Jones potential between two particles 'i' and 'j', and force on 'i' due
    to 'j'.

    Parameters
    ----------
    q_i : numpy.ndarray
        Coordinates vector of particle 'i' as a 1D-array of shape (n, ), where 'n' is the number of
        spatial dimensions. The calculated force will be for this particle.
    q_j : numpy.ndarray
        Coordinates vector of particle 'j' as a 1D-array of shape (n, ), where 'n' is the number of
        spatial dimensions.
    a_ij : float
        A-parameter of the Lennard-Jones potential for the two particles.
    b_ij : float
        B-parameter of the Lennard-Jones potential for the two particles.

    Returns
    -------
    force_i, potential_ij : tuple[numpy.ndarray, float]
        Force vector for particle 'i' due to particle 'j', followed by the potential energy between
        the two particles.

    Notes
    -----
    The force vector for particle 'j' due to particle 'i' is the same vector as the return value,
    only with opposite signs for all components, whereas the potential energy does not change.
    """
    # Calculate distance vector and its norm (i.e. distance)
    q_ji, d_ij = distances.two_arrays(q_i, q_j)
    # calculate potential
    e_ij = (a_ij / d_ij ** 12) - (b_ij / d_ij ** 6)
    # calculate force
    f_i = ((12 * a_ij / d_ij ** 14) - (6 * b_ij / d_ij ** 8)) * q_ji
    return f_i, e_ij


def bond_vibration_harmonic(
    q_i: np.ndarray, q_j: np.ndarray, d0: float, k_b: float
) -> Tuple[np.ndarray, float]:
    """
    Calculate the harmonic bond-vibration potential between two particles 'i' and 'j', and force on
    'i' due to 'j'.

    Parameters
    ----------
    q_i : numpy.ndarray
        Coordinates vector of particle 'i' as a 1D-array of shape (n, ), where 'n' is the number of
        spatial dimensions. The calculated force will be for this particle.
    q_j : numpy.ndarray
        Coordinates vector of particle 'j' as a 1D-array of shape (n, ), where 'n' is the number of
        spatial dimensions.
    d0 : float
        Equilibrium bond length.
    k_b : float
        Force constant of the harmonic bond potential.

    Returns
    -------
    force_i, potential_ij : tuple[numpy.ndarray, float]
        Force vector for particle 'i' due to particle 'j', followed by the potential energy between
        the two particles.

    Notes
    -----
    The force vector for particle 'j' due to particle 'i' is the same vector as the return value,
    only with opposite signs for all components, whereas the potential energy does not change.
    """
    # Calculate distance vector and its norm (i.e. distance)
    q_ji, d_ij = distances.two_arrays(q_i, q_j)
    # Calculate potential
    e_ij = 0.5 * k_b * (d_ij - d0) ** 2
    # Calculate force
    f_i = -k_b * (d_ij - d0) / d_ij * q_ji
    return f_i, e_ij


def angle_vibration_harmonic(
    q_i: np.ndarray, q_j: np.ndarray, q_k: np.ndarray, angle0: float, k_a: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Calculate the angle-vibration potential between three linearly bonded particles 'i', 'j' and
    'k' (where 'j' is the particle in the middle), and force on each one of them.

    Parameters
    ----------
    q_i : numpy.ndarray
        Coordinates vector of particle 'i' as a 1D-array of shape (n, ), where 'n' is the number of
        spatial dimensions.
    q_j : numpy.ndarray
        Coordinates vector of particle 'j' as a 1D-array of shape (n, ), where 'n' is the number of
        spatial dimensions. This is the particle in the middle.
    q_k : numpy.ndarray
        Coordinates vector of particle 'k' as a 1D-array of shape (n, ), where 'n' is the number of
        spatial dimensions.
    angle0 : float
        Equilibrium angle in radian.
    k_a : float
        Force constant of the harmonic angle potential.

    Returns
    -------
    force_i, force_j, force_k, potential_ijk :
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float]
        Force vector for each of the three particles in the order (i, j, k), followed by potential
        energy between the three particles.
    """
    # Calculate distance vectors and their norms (i.e. distances)
    q_ji, d_ij = distances.two_arrays(q_i, q_j)
    q_jk, d_jk = distances.two_arrays(q_k, q_j)
    # Calculate cosine of angle using the dot product formula
    cos = np.dot(q_ji, q_jk) / (d_ij * d_jk)
    # Raise error if cosine is not withing the range (-1, 1)
    if not (-1 < cos < 1):
        raise ValueError(f"Calculated cosine {cos} does not lie within the range (-1, 1).")
    # Calculate angle from cosine
    angle = np.arccos(cos)
    # Calculate potential
    e_ijk = 0.5 * k_a * (angle - angle0) ** 2
    # Calculate common term
    a = k_a * (angle - angle0) / abs(np.sin(angle))
    # Calculate forces
    f_i = a * (q_jk / (d_ij * d_jk) - cos * q_ji / d_ij ** 2)
    f_k = a * (q_ji / (d_ij * d_jk) - cos * q_jk / d_jk ** 2)
    f_j = a * ((-q_ji - q_jk) / (d_ij * d_jk) - cos * (-q_ji / d_ij ** 2 - q_jk / d_jk ** 2))
    return f_i, f_j, f_k, e_ijk


def dihedral():
    # TODO (not required for the water model)
    pass


def improper_dihedral():
    # TODO (not required for the water model)
    pass
