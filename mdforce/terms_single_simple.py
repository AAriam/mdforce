"""
Implementation of the individual terms of a general force field.

Each function calculates the force on a single target particle `i`, due to another single particle
`j` (except for `angle_vibration_harmonic`, which takes in three particles and calculates the force
on each of them). Moreover, each function also returns the potential energy of the system of
particles.

These functions are mostly intended for testing purposes; they are written in the most simple way
with the absolute minimal use of any other packages or modules, and their arguments are all numbers
, i.e. even the position vectors should each be inputted as three separate values corresponding to
the coordinates of the particle in x-, y- and z-directions.
"""

from typing import Tuple
import math  # Used only in `angle_vibration_harmonic` to calculate arccosine and sine.


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
    force_i_x, force_i_y, force_i_z, potential_ij : Tuple[float, float, float, float]
        Components of the force vector for particle 'i', in x-, y- and z-directions, respectively,
        followed by the potential energy between the two particles.

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
    dist_ji = (q_ji_x ** 2 + q_ji_y ** 2 + q_ji_z ** 2) ** 0.5
    # Calculate potential
    pot_ij = k_e * c_i * c_j / dist_ji
    # Calculate the components of the force vector
    common = k_e * c_i * c_j / dist_ji ** 3
    f_i_x = common * q_ji_x
    f_i_y = common * q_ji_y
    f_i_z = common * q_ji_z
    return f_i_x, f_i_y, f_i_z, pot_ij


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
    force_i_x, force_i_y, force_i_z, potential_ij : Tuple[float, float, float, float]
        Components of the force vector for particle 'i', in x-, y- and z-directions, respectively,
        followed by the potential energy between the two particles.

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
    dist_ji = (q_ji_x ** 2 + q_ji_y ** 2 + q_ji_z ** 2) ** 0.5
    # Calculate potential
    pot_ij = (a_ij / dist_ji ** 12) - (b_ij / dist_ji ** 6)
    # Calculate the components of the force vector
    repulsive_common = 12 * a_ij / dist_ji ** 14
    attractive_common = 6 * b_ij / dist_ji ** 8
    f_i_x = (repulsive_common * q_ji_x) - (attractive_common * q_ji_x)
    f_i_y = (repulsive_common * q_ji_y) - (attractive_common * q_ji_y)
    f_i_z = (repulsive_common * q_ji_z) - (attractive_common * q_ji_z)
    return f_i_x, f_i_y, f_i_z, pot_ij


def bond_vibration_harmonic(
    q_i_x: float,
    q_i_y: float,
    q_i_z: float,
    q_j_x: float,
    q_j_y: float,
    q_j_z: float,
    dist_eq: float,
    k_b: float,
) -> Tuple[float, float, float, float]:
    """
    Calculate the harmonic bond-vibration potential between two particles 'i' and 'j', and force on
    'i' due to 'j'.

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
    dist_eq : float
        Equilibrium bond length.
    k_b : float
        Force constant of the harmonic bond potential.

    Returns
    -------
    force_i_x, force_i_y, force_i_z, potential_ij : Tuple[float, float, float, float]
        Components of the force vector for particle 'i', in x-, y- and z-directions, respectively,
        followed by the potential energy between the two particles.

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
    dist_ji = (q_ji_x ** 2 + q_ji_y ** 2 + q_ji_z ** 2) ** 0.5
    # Calculate potential
    pot_ij = 0.5 * k_b * (dist_ji - dist_eq) ** 2
    # Calculate the components of the force vector
    common = -k_b * (dist_ji - dist_eq) / dist_ji
    f_i_x = common * q_ji_x
    f_i_y = common * q_ji_y
    f_i_z = common * q_ji_z
    return f_i_x, f_i_y, f_i_z, pot_ij


def angle_vibration_harmonic(
    q_j_x: float,
    q_j_y: float,
    q_j_z: float,
    q_i_x: float,
    q_i_y: float,
    q_i_z: float,
    q_k_x: float,
    q_k_y: float,
    q_k_z: float,
    angle_eq: float,
    k_a: float,
) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
    """
    Calculate the angle-vibration potential between three linearly bonded particles 'i', 'j' and
    'k' (where 'j' is the particle in the middle), and force on each one of them.

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
    angle_eq : float
        Equilibrium angle in radian.
    k_a : float
        Force constant of the harmonic angle potential.

    Returns
    -------
    f_i_x, f_i_y, f_i_z, f_j_x, f_j_y, f_j_z, f_k_x, f_k_y, f_k_z, potential_ijk :
    Tuple[float, float, float, float, float, float, float, float, float, float]
        Components of the force vector for particles 'i' 'j' and 'k', in x-, y- and z-directions,
        respectively, followed by the potential energy between the three particles.
    """
    # Calculate distance vector between 'j' and 'i'
    q_ji_x = q_i_x - q_j_x
    q_ji_y = q_i_y - q_j_y
    q_ji_z = q_i_z - q_j_z
    # Calculate distance vector between 'j' and 'k'
    q_jk_x = q_k_x - q_j_x
    q_jk_y = q_k_y - q_j_y
    q_jk_z = q_k_z - q_j_z
    # Calculate the norm of vectors
    dist_ji = (q_ji_x ** 2 + q_ji_y ** 2 + q_ji_z ** 2) ** 0.5
    dist_jk = (q_jk_x ** 2 + q_jk_y ** 2 + q_jk_z ** 2) ** 0.5
    # Calculate cosine of angle using the dot product formula
    cos = ((q_ji_x * q_jk_x) + (q_ji_y * q_jk_y) + (q_ji_z * q_jk_z)) / (dist_ji * dist_jk)
    # Raise error if cosine is not withing the range (-1, 1)
    if not (-1 < cos < 1):
        raise ValueError(f"Calculated cosine {cos} does not lie within the range (-1, 1).")
    # Calculate angle from cosine
    angle = math.acos(cos)
    # Calculate sine of angle
    sin = math.sin(angle)

    # Calculate potential
    pot_ijk = 0.5 * k_a * (angle - angle_eq) ** 2

    # Calculate common term
    common = k_a * (angle - angle_eq) / abs(sin)

    # Calculate the components of the force vector for 'i'
    f_i_x = common * (q_jk_x / (dist_ji * dist_jk) - cos * q_ji_x / dist_ji ** 2)
    f_i_y = common * (q_jk_y / (dist_ji * dist_jk) - cos * q_ji_y / dist_ji ** 2)
    f_i_z = common * (q_jk_z / (dist_ji * dist_jk) - cos * q_ji_z / dist_ji ** 2)
    # Calculate the components of the force vector for 'k'
    f_k_x = common * (q_ji_x / (dist_ji * dist_jk) - cos * q_jk_x / dist_jk ** 2)
    f_k_y = common * (q_ji_y / (dist_ji * dist_jk) - cos * q_jk_y / dist_jk ** 2)
    f_k_z = common * (q_ji_z / (dist_ji * dist_jk) - cos * q_jk_z / dist_jk ** 2)
    # Calculate the components of the force vector for 'j'
    f_j_x = common * (
        (-q_ji_x - q_jk_x) / (dist_ji * dist_jk)
        - cos * (-q_ji_x / dist_ji ** 2 - q_jk_x / dist_jk ** 2)
    )
    f_j_y = common * (
        (-q_ji_y - q_jk_y) / (dist_ji * dist_jk)
        - cos * (-q_ji_y / dist_ji ** 2 - q_jk_y / dist_jk ** 2)
    )
    f_j_z = common * (
        (-q_ji_z - q_jk_z) / (dist_ji * dist_jk)
        - cos * (-q_ji_z / dist_ji ** 2 - q_jk_z / dist_jk ** 2)
    )
    return f_j_x, f_j_y, f_j_z, f_i_x, f_i_y, f_i_z, f_k_x, f_k_y, f_k_z, pot_ijk


def dihedral():
    # TODO (not required for the water model)
    pass


def improper_dihedral():
    # TODO (not required for the water model)
    pass
