"""
Implementation of the individual terms of a general force field.

These functions are written in the most simple way with the absolute minimal use of any other
packages or modules, and are mostly for testing purposes.

Each function calculates the force on a single target particle `i`, due to another single particle
`j`, except for `angle_vibration`, which takes in three particles and calculates the force on each
of them. Moreover, each function also returns the potential energy of the system of two particles.
"""

from typing import Tuple
import math


def coulomb(
        q_i_x: float,
        q_i_y: float,
        q_i_z: float,
        q_j_x: float,
        q_j_y: float,
        q_j_z: float,
        c_i: float,
        c_j: float,
        k_e: float,
) -> Tuple[float, float, float, float]:
    """
    Calculate the coulomb potential between two particles 'i' and 'j', and force on 'i' due to 'j'.

    The formula for coulomb potential 'V_e' as a function of particles' position vectors 'q_i' and
    'q_j', with charges 'c_i' and 'c_j' is:
    V_e(q_i, q_j) = k_e * c_i * c_j / ||q_ji||
    where q_ji is the distance vector q_i - q_j = (q_ji_x, q_ji_y, q_ji_z)
    and the norm ||q_ji|| = sqrt(q_ji_x^2 + q_ji_y^2 + q_ji_z^2)

    Consequently, the force vector 'F_e' on particle 'i' due to particle 'j' is:
    F_e(i, j) = k_e * c_i * c_j * (q_ji) / ||q_ji||^3

    Parameters
    ----------
    q_i_x : float
        Coordinate of particle 'i' in x-direction.
    q_i_y : float
        Coordinate of particle 'i' in y-direction.
    q_i_z : float
        Coordinate of particle 'i' in z-direction.
    q_j_x : float
        Coordinate of particle 'j' in x-direction.
    q_j_y : float
        Coordinate of particle 'j' in y-direction.
    q_j_z : float
        Coordinate of particle 'j' in z-direction.
    c_i : float
        Charge of particle 'i'.
    c_j : float
        Charge of particle 'j'.
    k_e : float
        Coulomb constant, i.e. (1 / 4πε0).

    Returns
    -------
    v_ij, f_i_x, f_i_y, f_i_z : Tuple[float, float, float, float]
        Potential 'v_ij' between the two particles, followed by the components of the force vector
        for particle 'i', in x-, y- and z-directions, respectively.

    Notes
    -----
    The force vector for particle 'j' due to particle 'i' should be the same vector as the return
    value, only with opposite signs for all three components, whereas the potential should not
    change.
    """
    # Calculate distance vector
    q_ji_x = q_i_x - q_j_x
    q_ji_y = q_i_y - q_j_y
    q_ji_z = q_i_z - q_j_z
    # Calculate the norm of vector
    q_ji_norm = (q_ji_x ** 2 + q_ji_y ** 2 + q_ji_z ** 2) ** 0.5
    # Calculate potential
    v = k_e * c_i * c_j / q_ji_norm
    # Calculate the components of the force vector
    common = k_e * c_i * c_j / q_ji_norm ** 3
    f_i_x = common * q_ji_x
    f_i_y = common * q_ji_y
    f_i_z = common * q_ji_z
    return v, f_i_x, f_i_y, f_i_z


def lennard_jones(
        q_i_x: float,
        q_i_y: float,
        q_i_z: float,
        q_j_x: float,
        q_j_y: float,
        q_j_z: float,
        a_ij: float,
        b_ij: float,
) -> Tuple[float, float, float, float]:
    """
    Calculate the Lennard-Jones potential between two particles 'i' and 'j', and force on 'i' due
    to 'j'.

    Parameters
    ----------
    q_i_x : float
        Coordinate of particle 'i' in x-direction.
    q_i_y : float
        Coordinate of particle 'i' in y-direction.
    q_i_z : float
        Coordinate of particle 'i' in z-direction.
    q_j_x : float
        Coordinate of particle 'j' in x-direction.
    q_j_y : float
        Coordinate of particle 'j' in y-direction.
    q_j_z : float
        Coordinate of particle 'j' in z-direction.
    a_ij : float
        Lennard-Jones parameter A for the pair of particles.
    b_ij : float
        Lennard-Jones parameter B for the pair of particles.

    Returns
    -------
    v_ij, f_i_x, f_i_y, f_i_z : Tuple[float, float, float, float]
        Potential 'v_ij' between the two particles, followed by the components of the force vector
        for particle 'i', in x-, y- and z-directions, respectively.

    Notes
    -----
    The force vector for particle 'j' due to particle 'i' should be the same vector as the return
    value, only with opposite signs for all three components, whereas the potential should not
    change.
    """
    # Calculate distance vector
    q_ji_x = q_i_x - q_j_x
    q_ji_y = q_i_y - q_j_y
    q_ji_z = q_i_z - q_j_z
    # Calculate the norm of vector
    q_ji_norm = (q_ji_x ** 2 + q_ji_y ** 2 + q_ji_z ** 2) ** 0.5
    # Calculate potential
    v = (a_ij / q_ji_norm**12) - (b_ij / q_ji_norm**6)
    # Calculate the components of the force vector
    repulsive_common = 12 * a_ij / q_ji_norm**14
    attractive_common = 6 * b_ij / q_ji_norm**8
    f_i_x = (repulsive_common * q_ji_x) - (attractive_common * q_ji_x)
    f_i_y = (repulsive_common * q_ji_y) - (attractive_common * q_ji_y)
    f_i_z = (repulsive_common * q_ji_z) - (attractive_common * q_ji_z)
    return v, f_i_x, f_i_y, f_i_z


def bond_vibration_harmonic(
        q_i_x: float,
        q_i_y: float,
        q_i_z: float,
        q_j_x: float,
        q_j_y: float,
        q_j_z: float,
        k_b: float,
        q_eq: float,
) -> Tuple[float, float, float, float]:
    """
    Calculate the bond-vibration potential between two particles 'i' and 'j', and force on 'i' due
    to 'j'.

    Parameters
    ----------
    q_i_x : float
        Coordinate of particle 'i' in x-direction.
    q_i_y : float
        Coordinate of particle 'i' in y-direction.
    q_i_z : float
        Coordinate of particle 'i' in z-direction.
    q_j_x : float
        Coordinate of particle 'j' in x-direction.
    q_j_y : float
        Coordinate of particle 'j' in y-direction.
    q_j_z : float
        Coordinate of particle 'j' in z-direction.
    k_b : float
        Force constant of the bond.
    q_eq : float
        Equilibrium bond length.

    Returns
    -------
    v_ij, f_i_x, f_i_y, f_i_z : Tuple[float, float, float, float]
        Potential 'v_ij' between the two particles, followed by the components of the force vector
        for particle 'i', in x-, y- and z-directions, respectively.

    Notes
    -----
    The force vector for particle 'j' due to particle 'i' should be the same vector as the return
    value, only with opposite signs for all three components, whereas the potential should not
    change.
    """
    # Calculate distance vector
    q_ji_x = q_i_x - q_j_x
    q_ji_y = q_i_y - q_j_y
    q_ji_z = q_i_z - q_j_z
    # Calculate the norm of vector
    q_ji_norm = (q_ji_x ** 2 + q_ji_y ** 2 + q_ji_z ** 2) ** 0.5
    # Calculate potential
    v = 0.5 * k_b * (q_ji_norm - q_eq)**2
    # Calculate the components of the force vector
    common = -k_b * (q_ji_norm - q_eq) / q_ji_norm
    f_i_x = common * q_ji_x
    f_i_y = common * q_ji_y
    f_i_z = common * q_ji_z
    return v, f_i_x, f_i_y, f_i_z


def angle_vibration_harmonic(
        q_i_x: float,
        q_i_y: float,
        q_i_z: float,
        q_j_x: float,
        q_j_y: float,
        q_j_z: float,
        q_k_x: float,
        q_k_y: float,
        q_k_z: float,
        k_a: float,
        angle_eq: float,
) -> Tuple[float, float, float, float]:
    """
    Calculate the angle-vibration potential between three particles 'i', 'j' and 'k' (where 'j' is
    the particle in the middle), and force on each one of them.

    Parameters
    ----------
    q_i_x : float
        Coordinate of particle 'i' in x-direction.
    q_i_y : float
        Coordinate of particle 'i' in y-direction.
    q_i_z : float
        Coordinate of particle 'i' in z-direction.
    q_j_x : float
        Coordinate of particle 'j' in x-direction. This is the particle in the middle.
    q_j_y : float
        Coordinate of particle 'j' in y-direction. This is the particle in the middle.
    q_j_z : float
        Coordinate of particle 'j' in z-direction. This is the particle in the middle.
    q_k_x : float
        Coordinate of particle 'k' in x-direction.
    q_k_y : float
        Coordinate of particle 'k' in y-direction.
    q_k_z : float
        Coordinate of particle 'k' in z-direction.
    k_a : float
        Force constant of the angle.
    angle_eq : float
        Equilibrium angle in radian.

    Returns
    -------
    v_ijk, f_i_x, f_i_y, f_i_z, f_j_x, f_j_y, f_j_z, f_k_x, f_k_y, f_k_z :
    Tuple[float, float, float, float, float, float, float, float, float, float]
        Potential 'v_ijk' between the three particles, followed by the components of the force
        vector for particles 'i' 'j' and 'k', in x-, y- and z-directions, respectively.
    """
    # Calculate distance vectors
    q_ji_x = q_i_x - q_j_x
    q_ji_y = q_i_y - q_j_y
    q_ji_z = q_i_z - q_j_z

    q_jk_x = q_k_x - q_j_x
    q_jk_y = q_k_y - q_j_y
    q_jk_z = q_k_z - q_j_z

    # Calculate the norm of vectors
    q_ji_norm = (q_ji_x ** 2 + q_ji_y ** 2 + q_ji_z ** 2) ** 0.5
    q_jk_norm = (q_jk_x ** 2 + q_jk_y ** 2 + q_jk_z ** 2) ** 0.5

    # Calculate cosine of angle
    cos = ((q_ji_x * q_jk_x) + (q_ji_y * q_jk_y) + (q_ji_z * q_jk_z)) / (q_ji_norm * q_jk_norm)
    # Calculate angle from cosine
    angle = math.acos(cos)
    # Calculate sine of angle
    sin = math.sin(angle)

    # Calculate potential
    v = 0.5 * k_a * (angle - angle_eq)**2

    # Calculate common term
    common = k_a * (angle - angle_eq) / abs(sin)
    # Calculate the components of the force vector for 'i'
    f_i_x = common * (q_jk_x / (q_ji_norm * q_jk_norm) - cos * q_ji_x / q_ji_norm * 2)
    f_i_y = common * (q_jk_y / (q_ji_norm * q_jk_norm) - cos * q_ji_y / q_ji_norm * 2)
    f_i_z = common * (q_jk_z / (q_ji_norm * q_jk_norm) - cos * q_ji_z / q_ji_norm * 2)
    # Calculate the components of the force vector for 'k'
    f_k_x = common * (q_ji_x / (q_ji_norm * q_jk_norm) - cos * q_jk_x / q_jk_norm * 2)
    f_k_y = common * (q_ji_y / (q_ji_norm * q_jk_norm) - cos * q_jk_y / q_jk_norm * 2)
    f_k_z = common * (q_ji_z / (q_ji_norm * q_jk_norm) - cos * q_jk_z / q_jk_norm * 2)
    # Calculate the components of the force vector for 'j'
    f_j_x = common * ((  ))
    f_j_y = common * ()
    f_j_z = common * ()
    return v, f_i_x, f_i_y, f_i_z

