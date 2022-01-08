"""
Implementation of the individual terms of a general force field. Each function calculates the force
on a single target particle `q_i`, due to another single particle `q_j`, except for `angle_vibration`,
which takes in three particles and calculates the force on each of them.
Moreover, each function also returns the potential energy of the system of two particles.
"""

import numpy as np
import numpy.linalg as lin


def lennard_jones(
        q_i: np.ndarray,
        q_j: np.ndarray,
        a: float,
        b: float
) -> tuple[np.ndarray, float]:
    """
    Calculate the Lennard-Jones potential between two particles,
    and force on the first particle due to the second particle.

    Parameters
    ----------
    q_i : numpy.ndarray
        Coordinates of the target particle.
        The calculated force will be for this particle.
    q_j : numpy.ndarray
        Coordinates of the other particle.
    a : float
        A-parameter of the Lennard-Jones potential for the two particles.
    b : float
        B-parameter of the Lennard-Jones potential for the two particles.

    Returns
    -------
    force_vector, potential : tuple[numpy.ndarray, float]
        Force vector for the target particle,
        followed by potential energy of the system.

    Notes
    -----
    The force vector for the other particle due
    to the target particle will be the same vector
    as the return value, only with opposite sign.
    """

    # calculate common terms
    r_ji = q_i - q_j
    dist = lin.norm(r_ji)
    inverse_dist_2 = 1 / dist ** 2
    inverse_dist_6 = inverse_dist_2 ** 3

    # calculate potential using common terms
    e_attractive = -b * inverse_dist_6
    e_repulsive = a * inverse_dist_6 ** 2
    e = e_repulsive + e_attractive

    # calculate force using the calculated potential and common terms
    f_repulsive = 12 * e_repulsive
    f_attractive = 6 * e_attractive
    f_i = (f_attractive + f_repulsive) * inverse_dist_2 * r_ji
    return f_i, e


def coulomb(
        q_i: np.ndarray,
        q_j: np.ndarray,
        c_i: float,
        c_j: float,
        k: float
) -> tuple[np.ndarray, float]:
    """
    Calculate the Coulomb potential between two particles,
    and force on the first particle due to the second particle.

    Parameters
    ----------
    q_i : numpy.ndarray
        Coordinates of the target particle.
        The calculated force will be for this particle.
    q_j : numpy.ndarray
        Coordinates of the other particle.
    c_i : float
        Charge of the target particle.
    c_j : float
        Charge of the other particle.
    k : float
        Coulomb constant, i.e. (1 / 4πε0).

    Returns
    -------
    force_vector, potential : tuple[numpy.ndarray, float]
        Force vector for the target particle,
        followed by potential energy of the system.

    Notes
    -----
    The force vector for the other particle due
    to the target particle will be the same vector
    as the return value, only with opposite sign.
    """

    # Calculate common terms
    r_ji = q_i - q_j
    dist = lin.norm(r_ji)

    # Calculate potential
    e = k * c_i * c_j / dist

    # Calculate force
    f_i = e * r_ji / dist ** 2
    return f_i, e


def bond_vibration_harmonic(
        q_i: np.ndarray,
        q_j: np.ndarray,
        dist_eq: float,
        k: float
) -> tuple[np.ndarray, float]:
    """
    Calculate the harmonic bond-stretching potential between two bonded particles,
    and force on the first particle due to the second particle.

    Parameters
    ----------
    q_i : numpy.ndarray
        Coordinates of the target particle.
        The calculated force will be for this particle.
    q_j : numpy.ndarray
        Coordinates of the other particle.
    dist_eq : float
        Equilibrium bond length
    k : float
        Force constant of the harmonic potential.

    Returns
    -------
    force_vector, potential : tuple[numpy.ndarray, float]
        Force vector for the target particle,
        followed by potential energy of the system.

    Notes
    -----
    The force vector for the other particle due
    to the target particle will be the same vector
    as the return value, only with opposite sign.
    """

    # Calculate common terms
    r_ji = q_i - q_j
    dist = lin.norm(r_ji)
    displacement = dist - dist_eq

    # Calculate potential
    e = k * displacement ** 2 / 2

    # Calculate force
    f_i = (-k * displacement / dist) * r_ji
    return f_i, e


def angle_vibration(
        q_m: np.ndarray,
        q_l: np.ndarray,
        q_r: np.ndarray,
        angle_eq: float,
        k: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Calculate the harmonic angle-vibration potential between three linearly bonded particles,
    and force on each of the three particles.

    Parameters
    ----------
    q_m : numpy.ndarray
        Coordinates of the particle in the middle of the triplet.
    q_l : numpy.ndarray
        Coordinates of the particle on one end of the triplet.
    q_r : numpy.ndarray
        Coordinates of the particle on the other end of the triplet.
    angle_eq : float
        Equilibrium angle in radian.
    k : float
        Force constant of the harmonic potential.

    Returns
    -------
    force_vect_m, force_vect_l, force_vect_r, potential : tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float]
        Force vector for each of the three particles in the order (mid, left, right),
        followed by potential energy of the system.
    """

    # calculate the angle theta
    r_ml = q_l - q_m
    r_mr = q_r - q_m
    dist_ml = lin.norm(r_ml)
    dist_mr = lin.norm(r_mr)
    cos = np.dot(r_ml, r_mr) / (dist_ml * dist_mr)
    # due to floating point calculation errors, sometimes
    # cos will be slightly greater than 1 or smaller than -1,
    # in which case the angle cannot be calculated by the arccos.
    # Thus check the value of cos, and correct it if necessary.
    if 1 > cos > -1:
        pass
    elif np.isclose(cos, 1):
        cos = 1
    elif np.isclose(cos, -1):
        cos = -1
    else:
        raise ValueError(f"Calculated cosine {cos} does not lie within the range of [-1, 1].")
    angle = np.arccos(cos)

    # Calculate common terms
    sin = np.sin(angle)
    angle_displacement = angle - angle_eq
    a = k * angle_displacement / sin
    norm_mult_lm_rm = dist_ml * dist_mr

    # Calculate potential
    e = k * angle_displacement ** 2

    # Calculate f_l
    b = r_mr / norm_mult_lm_rm
    c = cos * r_ml / dist_ml ** 2
    f_l = a * (b - c)

    # Calculate f_r
    b2 = r_ml / norm_mult_lm_rm
    c2 = cos * r_mr / dist_mr ** 2
    f_r = a * (b2 - c2)

    # Calculate f_m
    b3 = (2 * q_m - q_l - q_r) / norm_mult_lm_rm
    c3 = cos * (r_ml / dist_ml ** 2 + r_mr / dist_mr ** 2)
    f_m = a * (b3 + c3)
    return f_m, f_l, f_r, e


def dihedral():
    # TODO (not required for the water model)
    pass


def improper_dihedral():
    # TODO (not required for the water model)
    pass
