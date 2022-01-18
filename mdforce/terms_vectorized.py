"""
Implementation of the individual terms of a general force field in a vectorized fashion.
Each function calculates the force on a single target particle `q_i`, due to a number of
other particles `q_js`, except for `angle_vibration_harmonic`, which takes in three particles
and calculates the force on each of them.
Moreover, each function also returns the potential energy of the whole system of particles.
"""

# Standard library
from typing import Tuple

# 3rd-party
import numpy as np
import numpy.linalg as lin


def lennard_jones(
        q_i: np.ndarray,
        q_js: np.ndarray,
        a_ijs: np.ndarray,
        b_ijs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate the Lennard-Jones potential and the force between
    one particle and a number of other particles.

    Parameters
    ----------
    q_i : numpy.ndarray
        1D-array containing the coordinates of the target particle.
    q_js : numpy.ndarray
        2D-array containing the coordinates of other particles.
    a_ijs : float
        1D-array containing the A-parameters of the potential between `q_i` and each particle in `q_js`.
    b_ijs : float
        1D-array containing the B-parameters of the potential between `q_i` and each particle in `q_js`.

    Returns
    -------
    f_i_total, f_js, e_total : Tuple[numpy.ndarray, numpy.ndarray, float]
        f_i_total: Total force-vector on `q_i`, as a 1D-array with same size as `q_i`.
        f_js: Force-vector on each particle in `q_js` due to `q_i`, as a 2D-array with same shape as `q_js`.
        e_total: Total potential energy between `q_i` and all the particles in `q_js`.
    """

    # Calculate common terms
    r_jsi = q_i - q_js
    dist = lin.norm(r_jsi, axis=1)
    inverse_dist_2 = 1 / dist ** 2
    inverse_dist_6 = inverse_dist_2 ** 3

    # Calculate potential
    e_attractive = -b_ijs * inverse_dist_6
    e_repulsive = a_ijs * inverse_dist_6 ** 2
    e_total = (e_repulsive + e_attractive).sum()

    # Calculate force
    f_attractive = 6 * e_attractive
    f_repulsive = 12 * e_repulsive
    f_i = ((f_attractive + f_repulsive) * inverse_dist_2).reshape(-1, 1) * r_jsi
    f_js = -f_i
    f_i_total = f_i.sum(axis=0)
    return f_i_total, f_js, e_total


def coulomb(
        q_i: np.ndarray,
        q_js: np.ndarray,
        c_i: float,
        c_js: np.ndarray,
        k: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate the Coulomb potential and the force between
    one particle and a number of other particles.

    Parameters
    ----------
    q_i : numpy.ndarray
        1D-array containing the coordinates of the target particle.
    q_js : numpy.ndarray
        2D-array containing the coordinates of other particles.
    c_i : float
        Charge of the target particle 'i'.
    c_js : numpy.ndarray
        Charge of each particle in `q_js`.
    k : float
        Coulomb constant, i.e. (1 / 4πε0).

    Returns
    -------
    f_i_total, f_js, e_total : Tuple[numpy.ndarray, numpy.ndarray, float]
        f_i_total: Total force-vector on `q_i`, as a 1D-array with same size as `q_i`.
        f_js: Force-vector on each particle in `q_js` due to `q_i`, as a 2D-array with same shape as `q_js`.
        e_total: Total potential energy between `q_i` and all the particles in `q_js`.
    """

    # Calculate common terms
    r_jsi = q_i - q_js
    dist = lin.norm(r_jsi, axis=1)

    # Calculate potential
    e = k * c_i * c_js / dist
    e_total = e.sum()

    # Calculate force
    f_i = (e / dist ** 2).reshape(-1, 1) * r_jsi
    f_js = -f_i
    f_i_total = f_i.sum(axis=0)
    return f_i_total, f_js, e_total


def bond_vibration_harmonic(
        q_i: np.ndarray,
        q_js: np.ndarray,
        eq_dist_ijs: np.ndarray,
        k_ijs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate the bond vibration potential and the force between
    one particle and a number of other bonded particles.

    Parameters
    ----------
    q_i : numpy.ndarray
        1D-array containing the coordinates of the target particle.
    q_js : numpy.ndarray
        2D-array containing the coordinates of other bonded particles.
    eq_dist_ijs : numpy.ndarray
        1D-array containing the equilibrium bond lengths between particle 'i' and each particle in `q_js`.
    k_ijs : numpy.ndarray
        1D-array containing the force constant of the harmonic potential
        between particle 'i' and each particle in `q_js`.

    Returns
    -------
    f_i_total, f_js, e_total : Tuple[numpy.ndarray, numpy.ndarray, float]
        f_i_total: Total force-vector on `q_i`, as a 1D-array with same size as `q_i`.
        f_js: Force-vector on each particle in `q_js` due to `q_i`, as a 2D-array with same shape as `q_js`.
        e_total: Total potential energy between `q_i` and all the particles in `q_js`.
    """

    # Calculate common terms
    r_jsi = q_i - q_js
    dist = lin.norm(r_jsi, axis=1)
    displacement = dist - eq_dist_ijs
    k_times_displ = k_ijs * displacement

    # Calculate potential
    e_total = (k_times_displ * displacement / 2).sum()

    # Calculate force
    f_i = (-k_times_displ / dist).reshape(-1, 1) * r_jsi
    f_js = -f_i
    f_i_total = f_i.sum(axis=0)
    return f_i_total, f_js, e_total


def angle_vibration_harmonic():
    # TODO
    pass


def dihedral():
    # TODO
    pass


def improper_dihedral():
    # TODO
    pass
