"""
Implementation of the individual terms of a general force field.

Each function calculates the force on a single target particle `i`, due to another single particle
`j` (except for `angle_vibration_harmonic`, which takes in three particles and calculates the force
on each of them). Moreover, each function also returns the potential energy of the system of
particles.

These functions are mostly intended for testing purposes; They are one step more complicated than
the equivalent functions in module `terms_single_simple`, and accept the position vector arguments
as numpy arrays.
"""

# Standard library
from typing import Tuple

# 3rd-party
import numpy as np
import numpy.linalg as lin


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
    The force vector for particle 'j' due to particle 'i' will be the same vector as the return
    value, only with opposite sign, whereas the potential will not change.
    """
    # Calculate common terms
    q_ji = q_i - q_j
    dist_ji = lin.norm(q_ji)
    # Calculate potential
    pot_ij = k_e * c_i * c_j / dist_ji
    # Calculate force
    f_i = pot_ij * q_ji / dist_ji ** 2
    return f_i, pot_ij


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
    The force vector for particle 'j' due to particle 'i' will be the same vector as the return
    value, only with opposite sign, whereas the potential will not change.
    """
    # calculate common terms
    q_ji = q_i - q_j
    dist_ji = lin.norm(q_ji)
    inverse_dist_2 = 1 / dist_ji ** 2
    inverse_dist_6 = inverse_dist_2 ** 3
    # calculate potential using common terms
    pot_ij_attractive = -b_ij * inverse_dist_6
    pot_ij_repulsive = a_ij * inverse_dist_6 ** 2
    pot_ij = pot_ij_repulsive + pot_ij_attractive
    # calculate force using the calculated potential and common terms
    f_repulsive = 12 * pot_ij_repulsive
    f_attractive = 6 * pot_ij_attractive
    f_i = (f_attractive + f_repulsive) * inverse_dist_2 * q_ji
    return f_i, pot_ij


def bond_vibration_harmonic(
    q_i: np.ndarray, q_j: np.ndarray, dist_eq: float, k_b: float
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
    dist_eq : float
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
    The force vector for particle 'j' due to particle 'i' will be the same vector as the return
    value, only with opposite sign, whereas the potential will not change.
    """
    # Calculate common terms
    q_ji = q_i - q_j
    dist_ji = lin.norm(q_ji)
    displacement = dist_ji - dist_eq
    k_times_displ = k_b * displacement
    # Calculate potential
    pot_ij = k_times_displ * displacement / 2
    # Calculate force
    f_i = (-k_times_displ / dist_ji) * q_ji
    return f_i, pot_ij


def angle_vibration_harmonic(
    q_j: np.ndarray, q_i: np.ndarray, q_k: np.ndarray, angle_eq: float, k_a: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Calculate the angle-vibration potential between three linearly bonded particles 'i', 'j' and
    'k' (where 'j' is the particle in the middle), and force on each one of them.

    Parameters
    ----------
    q_j : numpy.ndarray
        Coordinates vector of particle 'j' as a 1D-array of shape (n, ), where 'n' is the number of
        spatial dimensions. This is the particle in the middle of the triplet.
    q_i : numpy.ndarray
        Coordinates vector of particle 'i' as a 1D-array of shape (n, ), where 'n' is the number of
        spatial dimensions.
    q_k : numpy.ndarray
        Coordinates vector of particle 'k' as a 1D-array of shape (n, ), where 'n' is the number of
        spatial dimensions.
    angle_eq : float
        Equilibrium angle in radian.
    k_a : float
        Force constant of the harmonic angle potential.

    Returns
    -------
    force_j, force_i, force_k, potential_ijk :
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float]
        Force vector for each of the three particles in the order (j, i, k), followed by potential
        energy between the three particles.
    """

    # Calculate distance vectors
    q_ji = q_i - q_j
    q_jk = q_k - q_j
    # Calculate the norm of vectors
    dist_ji = lin.norm(q_ji)
    dist_jk = lin.norm(q_jk)
    # Calculate cosine of angle using the dot product formula
    cos = np.dot(q_ji, q_jk) / (dist_ji * dist_jk)
    # Raise error if cosine is not withing the range (-1, 1)
    if not (-1 < cos < 1):
        raise ValueError(f"Calculated cosine {cos} does not lie within the range (-1, 1).")
    # Calculate angle from cosine
    angle = np.arccos(cos)

    # Calculate common terms
    sin = np.sin(angle)
    angle_displacement = angle - angle_eq
    a = k_a * angle_displacement / sin
    dist_ji_mult_dist_jk = dist_ji * dist_jk
    q_ji_div_dist2_ji = q_ji / dist_ji ** 2
    q_jk_div_dist2_jk = q_jk / dist_jk ** 2

    # Calculate potential
    pot_ijk = 0.5 * k_a * angle_displacement ** 2

    # Calculate force for particle 'i'
    b_i = q_jk / dist_ji_mult_dist_jk
    c_i = cos * q_ji_div_dist2_ji
    f_i = a * (b_i - c_i)

    # Calculate force for particle 'k'
    b_k = q_ji / dist_ji_mult_dist_jk
    c_k = cos * q_jk_div_dist2_jk
    f_k = a * (b_k - c_k)

    # Calculate force for particle 'j' (the middle particle)
    b_j = -(q_ji + q_jk) / dist_ji_mult_dist_jk
    c_j = cos * (q_ji_div_dist2_ji + q_jk_div_dist2_jk)
    f_j = a * (b_j + c_j)
    return f_j, f_i, f_k, pot_ijk


def dihedral():
    # TODO (not required for the water model)
    pass


def improper_dihedral():
    # TODO (not required for the water model)
    pass
