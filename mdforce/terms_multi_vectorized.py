"""
Implementation of the individual terms of a general force field.

Each function calculates the force for a given number of particles in a vectorized fashion; they
take in an array of coordinates for a single 'target' particle `i`, and an array of coordinates
for some other particles `js`, and calculate the force between `i` and each particle in `js`.
The return values will then be the total force on `i` due to all other particles in `js`, plus
an array of forces on each particle in `js` due to `i`. An exception is the function
`angle_vibration_harmonic`, which takes three particles and calculates the force on each of them.
Moreover, each function also returns the potential energy of each particle-pair/triplet in another
array.
"""

# Standard library
from typing import Tuple

# 3rd-party
import numpy as np
import numpy.linalg as lin


def coulomb(
    q_i: np.ndarray, q_js: np.ndarray, c_i: float, c_js: np.ndarray, k_e: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate the coulomb potential between a number of particle-pairs, and the total force on a
    single target particle 'i' due to a number of other particles 'js', and the force on each
    particle in 'js' due to particle 'i'.

    Parameters
    ----------
    q_i : numpy.ndarray
        Coordinates vector of the target particle 'i' as a 1D-array of shape (m, ), where 'm' is
        the number of spatial dimensions.
    q_js : numpy.ndarray
        Coordinates vectors of all interacting particles 'js' as a 2D-array of shape (n, m), where
        'n' is the number of particles, and 'm' is the number of spatial dimensions.
    c_i : float
        Charge of the target particle 'i'.
    c_js : numpy.ndarray
        Charges of all interacting particles `js`, as a 1D-array of shape (n, ). The value at each
        index corresponds to the charge of the particle at the same index in `q_js`.
    k_e : float
        Coulomb constant, i.e. (1 / 4πε0).

    Returns
    -------
    f_i_total, f_jsi, pot_ijs : Tuple[numpy.ndarray, numpy.ndarray, float]
        f_i_total: Total force-vector on particle 'i' due to all particles 'js', as a 1D-array with
        same shape as `q_i`.
        f_jsi: Force-vector on each particle in 'js' due to 'i', as a 2D-array with same shape as
        `q_js`.
        pot_ijs: Potential energy between 'i' and each particle in 'js', as a 1D-array of shape
        (n, ).
    """
    # Calculate common terms
    q_jsi = q_i - q_js
    dist_jsi = lin.norm(q_jsi, axis=1)
    # Calculate potential
    pot_ijs = k_e * c_i * c_js / dist_jsi
    # Calculate force
    f_ijs = (pot_ijs / dist_jsi ** 2).reshape(-1, 1) * q_jsi
    f_jsi = -f_ijs
    f_i_total = f_ijs.sum(axis=0)
    return f_i_total, f_jsi, pot_ijs


def lennard_jones(
    q_i: np.ndarray, q_js: np.ndarray, a_ijs: np.ndarray, b_ijs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate the Lennard-Jones potential between a number of particle-pairs, and the total force
    on a single target particle 'i' due to a number of other particles 'js', and the force on each
    particle in 'js' due to particle 'i'.

    Parameters
    ----------
    q_i : numpy.ndarray
        Coordinates vector of the target particle 'i' as a 1D-array of shape (m, ), where 'm' is
        the number of spatial dimensions.
    q_js : numpy.ndarray
        Coordinates vectors of all interacting particles 'js' as a 2D-array of shape (n, m), where
        'n' is the number of particles, and 'm' is the number of spatial dimensions.
    a_ijs : float
        A-parameters of the potential between 'i' and each interacting particle in `js`, as a
        1D-array of shape (n, ). The value at each index corresponds to the A-parameter between 'i'
        and the particle at the same index in `q_js`.
    b_ijs : float
        B-parameters of the potential between 'i' and each interacting particle in `js`, as a
        1D-array of shape (n, ). The value at each index corresponds to the B-parameter between 'i'
        and the particle at the same index in `q_js`.

    Returns
    -------
    f_i_total, f_jsi, pot_ijs : Tuple[numpy.ndarray, numpy.ndarray, float]
        f_i_total: Total force-vector on particle 'i' due to all particles 'js', as a 1D-array with
        same shape as `q_i`.
        f_jsi: Force-vector on each particle in 'js' due to 'i', as a 2D-array with same shape as
        `q_js`.
        pot_ijs: Potential energy between 'i' and each particle in 'js', as a 1D-array of shape
        (n, ).
    """
    # Calculate common terms
    q_jsi = q_i - q_js
    dist_jsi = lin.norm(q_jsi, axis=1)
    inverse_dist_2 = 1 / dist_jsi ** 2
    inverse_dist_6 = inverse_dist_2 ** 3
    # Calculate potential
    pot_ijs_attractive = -b_ijs * inverse_dist_6
    pot_ijs_repulsive = a_ijs * inverse_dist_6 ** 2
    pot_ijs = pot_ijs_repulsive + pot_ijs_attractive
    # Calculate force
    f_ijs_attractive = 6 * pot_ijs_attractive
    f_ijs_repulsive = 12 * pot_ijs_repulsive
    f_ijs = ((f_ijs_attractive + f_ijs_repulsive) * inverse_dist_2).reshape(-1, 1) * q_jsi
    f_jsi = -f_ijs
    f_i_total = f_ijs.sum(axis=0)
    return f_i_total, f_jsi, pot_ijs


def bond_vibration_harmonic(
    q_i: np.ndarray, q_js: np.ndarray, dist_eq_ijs: np.ndarray, k_b_ijs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate the harmonic bond-vibration potential between a number of particle-pairs, and the
    total force on a single target particle 'i' due to a number of other particles 'js', and the
    force on each particle in 'js' due to particle 'i'.

    Parameters
    ----------
    q_i : numpy.ndarray
        Coordinates vector of the target particle 'i' as a 1D-array of shape (m, ), where 'm' is
        the number of spatial dimensions.
    q_js : numpy.ndarray
        Coordinates vectors of all interacting particles 'js' as a 2D-array of shape (n, m), where
        'n' is the number of particles, and 'm' is the number of spatial dimensions.
    dist_eq_ijs : numpy.ndarray
        Equilibrium bond length between 'i' and each interacting particle in `js`, as a 1D-array of
        shape (n, ). The value at each index corresponds to the equilibrium bond length between 'i'
        and the particle at the same index in `q_js`.
    k_b_ijs : numpy.ndarray
        Force constant of the harmonic bond potential between 'i' and each interacting particle in
        `js`, as a 1D-array of shape (n, ). The value at each index corresponds to the force
        constant between 'i' and the particle at the same index in `q_js`.

    Returns
    -------
    f_i_total, f_jsi, pot_ijs : Tuple[numpy.ndarray, numpy.ndarray, float]
        f_i_total: Total force-vector on particle 'i' due to all particles 'js', as a 1D-array with
        same shape as `q_i`.
        f_jsi: Force-vector on each particle in 'js' due to 'i', as a 2D-array with same shape as
        `q_js`.
        pot_ijs: Potential energy between 'i' and each particle in 'js', as a 1D-array of shape
        (n, ).
    """
    # Calculate common terms
    q_jsi = q_i - q_js
    dist_jsi = lin.norm(q_jsi, axis=1)
    displacements_jsi = dist_jsi - dist_eq_ijs
    k_times_displacements = k_b_ijs * displacements_jsi
    # Calculate potential
    pot_ijs = k_times_displacements * displacements_jsi / 2
    # Calculate force
    f_ijs = (-k_times_displacements / dist_jsi).reshape(-1, 1) * q_jsi
    f_jsi = -f_ijs
    f_i_total = f_ijs.sum(axis=0)
    return f_i_total, f_jsi, pot_ijs


def angle_vibration_harmonic():
    # TODO
    pass


def dihedral():
    # TODO
    pass


def improper_dihedral():
    # TODO
    pass
