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
    f_i_total, f_jsi, e_ijs : Tuple[numpy.ndarray, numpy.ndarray, float]
        f_i_total: Total force-vector on particle 'i' due to all particles 'js', as a 1D-array with
        same shape as `q_i`.
        f_jsi: Force-vector on each particle in 'js' due to 'i', as a 2D-array with same shape as
        `q_js`.
        e_ijs: Potential energy between 'i' and each particle in 'js', as a 1D-array of shape
        (n, ).
    """
    # Calculate distance vectors
    q_jsi = q_i - q_js
    # Calculate the norm of vectors (i.e. distances)
    d_ijs = np.linalg.norm(q_jsi, axis=1)
    # Calculate potentials
    e_ijs = k_e * c_i * c_js / d_ijs
    # Calculate forces
    f_ijs = (e_ijs / d_ijs ** 2).reshape(-1, 1) * q_jsi
    return f_ijs.sum(axis=0), -f_ijs, e_ijs


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
    f_i_total, f_jsi, e_ijs : Tuple[numpy.ndarray, numpy.ndarray, float]
        f_i_total: Total force-vector on particle 'i' due to all particles 'js', as a 1D-array with
        same shape as `q_i`.
        f_jsi: Force-vector on each particle in 'js' due to 'i', as a 2D-array with same shape as
        `q_js`.
        e_ijs: Potential energy between 'i' and each particle in 'js', as a 1D-array of shape
        (n, ).
    """
    # Calculate distance vectors
    q_jsi = q_i - q_js
    # Calculate the norm of vectors (i.e. distances)
    d_ijs = np.linalg.norm(q_jsi, axis=1)
    # Calculate common terms
    inv_d_2 = 1 / d_ijs ** 2
    inv_d_6 = inv_d_2 ** 3
    # Calculate potentials
    e_ijs_repulsive = a_ijs * inv_d_6 ** 2
    e_ijs_attractive = -b_ijs * inv_d_6
    e_ijs = e_ijs_repulsive + e_ijs_attractive
    # Calculate forces
    f_ijs = (6 * (e_ijs + e_ijs_repulsive) * inv_d_2).reshape(-1, 1) * q_jsi
    return f_ijs.sum(axis=0), -f_ijs, e_ijs


def bond_vibration_harmonic(
    q_i: np.ndarray, q_js: np.ndarray, d0_ijs: np.ndarray, k_b_ijs: np.ndarray
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
    d0_ijs : numpy.ndarray
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
    # Calculate distance vectors
    q_jsi = q_i - q_js
    # Calculate the norm of vectors (i.e. distances)
    d_ijs = np.linalg.norm(q_jsi, axis=1)
    # Calculate common terms
    delta_d_ijs = d_ijs - d0_ijs
    k__delta_d_ijs = k_b_ijs * delta_d_ijs
    # Calculate potentials
    pot_ijs = k__delta_d_ijs * delta_d_ijs / 2
    # Calculate forces
    f_ijs = (-k__delta_d_ijs / d_ijs).reshape(-1, 1) * q_jsi
    return f_ijs.sum(axis=0), -f_ijs, pot_ijs


def angle_vibration_harmonic(
    q_is: np.ndarray,
    q_j: np.ndarray,
    q_ks: np.ndarray,
    angle0_isjks: np.ndarray,
    k_a_isjks: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # TODO
    pass


def dihedral():
    # TODO
    pass


def improper_dihedral():
    # TODO
    pass
