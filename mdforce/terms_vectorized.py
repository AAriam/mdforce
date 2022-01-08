"""
Implementation of the individual terms of the force field.
"""

import numpy as np
import numpy.linalg as lin


def lennard_jones(q_i, q_j, a, b):
    """
    Calculate the Lennard-Jones force between two particles.

    Parameters
    ----------
    q_i : numpy.ndarray
        Coordinates of the target particle.
    q_j : numpy.ndarray
        Coordinates of the other particle.
    a : float
        A-parameter of the potential.
    b : float
        B-parameter of the potential

    Returns
    -------
        numpy.ndarray
        Force vector for the target particle.

    Notes
    -----
    The force vector for the other particle
    will be the same vector as the return value,
    only with opposite sign.
    """
    q_ij = q_i - q_j
    q_ij_norm = lin.norm(q_i - q_j)
    repulsive = (12 * a / q_ij_norm ** 14) * q_ij
    attractive = (-6 * b / q_ij_norm ** 8) * q_ij
    f_qi = repulsive + attractive
    return f_qi


def coulomb(q, pair_idx, c, k):
    """
    Calculate the total Coulomb potential and force
    for a number of particles, due to other particles.

    Parameters
    ----------
    q : numpy.ndarray
        2D array of shape (n, m), containing the coordinates
        of n particles in an m-dimensional space.
    pair_idx : numpy.ndarray
        2D array of shape (p, 2), containing the indices
        of all pairs of particles in q, between which
        the Coulomb interaction should be calculated.
    c : numpy.ndarray
        1D array of length (n), containing the
        charges of all particles in q.
    k : float
        Coulomb constant, i.e. (1 / 4πε0).

    Returns
    -------
        tuple [numpy.ndarray, numpy.ndarray]
        Force vectors for all particles, as a 2D array of shape (n, m)
        followed by potential energy of all particles, as a 1D array of length (n).

    Notes
    -----
    For particles in q whose index is not in pair_idx,
    the respective positions in returned force and potential arrays
    will be zero.
    """

    f = np.zeros_like(q)
    e = np.zeros(q.shape[0])

    i = pair_idx[:, 0]
    j = pair_idx[:, 1]
    c_i = c[:, 0]
    c_j = c[:, 1]

    q_i = q[i]
    q_j = q[j]
    q_ij = q_i - q_j
    q_ij_norm = lin.norm(q_ij, axis=1).reshape(-1, 1)

    e = k * c_i * c_j / q_ij_norm


    return f, e


def bond_vibration_harmonic(q_i, q_j, eq_len, k):
    """
    Calculate the bond stretching force between two bonded particles,
    using the harmonic oscillator model.

    Parameters
    ----------
    q_i : numpy.ndarray
        Coordinates of the target particle.
    q_j : numpy.ndarray
        Coordinates of the other particle.
    eq_len : float
        Equilibrium bond length
    k : float
        Force constant of the harmonic potential.

    Returns
    -------
        numpy.ndarray
        Force vector for the target particle.

    Notes
    -----
    The force vector for the other particle
    will be the same vector as the return value,
    only with opposite sign.
    """
    q_ij_norm = lin.norm(q_i - q_j)
    f_q_i = (k * (q_ij_norm - eq_len) / q_ij_norm) * q_i
    return f_q_i


def angle_vibration(q_left, q_mid, q_right, theta_eq, k):

    # calculate the angle theta
    q_rm = q_right - q_mid
    q_lm = q_left - q_mid
    q_lm_norm = lin.norm(q_lm)
    q_rm_norm = lin.norm(q_rm)
    cos = np.inner(q_lm, q_rm) / (q_lm_norm * q_rm_norm)
    theta = np.arccos(cos)
    sin = np.sin(theta)

    a = k * (theta - theta_eq)

    b = q_rm / (q_lm_norm * q_rm_norm)
    c = cos * q_lm / (q_lm_norm ** 2 * q_rm_norm)
    f_q_left = a * (b - c) / sin

    b2 = q_lm / (q_lm_norm * q_rm_norm)
    c2 = cos * q_rm / (q_rm_norm ** 2 * q_lm_norm)
    f_q_right = a * (b2 - c2) / sin

    b3 = (q_left + q_right - 2 * q_mid) / (q_lm_norm * q_rm_norm)
    c3 = cos * (q_lm / q_lm_norm ** 2 + q_rm / q_rm_norm ** 2)
    f_q_mid = a * (b3 - c3) / sin
    return f_q_left, f_q_mid, f_q_right


def dihedral():
    pass


def improper_dihedral():
    pass
