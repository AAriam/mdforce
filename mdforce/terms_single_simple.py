"""
Implementation of the individual terms of a general force field.

Each function calculates the force on a single target particle `i`, due to another single particle
`j` (except for `angle_vibration_harmonic`, which takes in three particles and calculates the force
on each of them). Moreover, each function also returns the potential energy of the system of
particles.

These functions are mostly intended for testing purposes; they are written in the most primitive
way, with the absolute minimal use of any other packages or modules, and their arguments are all
numbers, i.e. even the position vectors should each be inputted as three separate values
corresponding to the coordinates of the particle in x-, y- and z-directions.
"""


# Standard library
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
    The force vector for particle 'j' due to particle 'i' is the same vector as the return value,
    only with opposite signs for all components, whereas the potential energy does not change.
    """
    # Calculate distance vector
    q_ji_x = q_i_x - q_j_x
    q_ji_y = q_i_y - q_j_y
    q_ji_z = q_i_z - q_j_z
    # Calculate the norm of vector (i.e. distance)
    d_ij = (q_ji_x ** 2 + q_ji_y ** 2 + q_ji_z ** 2) ** 0.5
    # Calculate potential
    e_ij = k_e * c_i * c_j / d_ij
    # Calculate the components of the force vector
    common = k_e * c_i * c_j / d_ij ** 3
    f_i_x = common * q_ji_x
    f_i_y = common * q_ji_y
    f_i_z = common * q_ji_z
    return f_i_x, f_i_y, f_i_z, e_ij


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
    The force vector for particle 'j' due to particle 'i' is the same vector as the return value,
    only with opposite signs for all components, whereas the potential energy does not change.
    """
    # Calculate distance vector
    q_ji_x = q_i_x - q_j_x
    q_ji_y = q_i_y - q_j_y
    q_ji_z = q_i_z - q_j_z
    # Calculate the norm of vector (i.e. distance)
    d_ij = (q_ji_x ** 2 + q_ji_y ** 2 + q_ji_z ** 2) ** 0.5
    # Calculate potential
    e_ij = (a_ij / d_ij ** 12) - (b_ij / d_ij ** 6)
    # Calculate the components of the force vector
    f_repulsive_common = 12 * a_ij / d_ij ** 14
    f_attractive_common = 6 * b_ij / d_ij ** 8
    f_i_x = (f_repulsive_common * q_ji_x) - (f_attractive_common * q_ji_x)
    f_i_y = (f_repulsive_common * q_ji_y) - (f_attractive_common * q_ji_y)
    f_i_z = (f_repulsive_common * q_ji_z) - (f_attractive_common * q_ji_z)
    return f_i_x, f_i_y, f_i_z, e_ij


def bond_vibration_harmonic(
    q_i_x: float,
    q_i_y: float,
    q_i_z: float,
    q_j_x: float,
    q_j_y: float,
    q_j_z: float,
    d0: float,
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
    d0 : float
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
    The force vector for particle 'j' due to particle 'i' is the same vector as the return value,
    only with opposite signs for all components, whereas the potential energy does not change.
    """
    # Calculate distance vector
    q_ji_x = q_i_x - q_j_x
    q_ji_y = q_i_y - q_j_y
    q_ji_z = q_i_z - q_j_z
    # Calculate the norm of vector (i.e. distance)
    d_ij = (q_ji_x ** 2 + q_ji_y ** 2 + q_ji_z ** 2) ** 0.5
    # Calculate potential
    e_ij = 0.5 * k_b * (d_ij - d0) ** 2
    # Calculate the components of the force vector
    common = -k_b * (d_ij - d0) / d_ij
    f_i_x = common * q_ji_x
    f_i_y = common * q_ji_y
    f_i_z = common * q_ji_z
    return f_i_x, f_i_y, f_i_z, e_ij


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
    angle0: float,
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
    angle0 : float
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
    # Calculate the norm of vectors (i.e. distances)
    d_ij = (q_ji_x ** 2 + q_ji_y ** 2 + q_ji_z ** 2) ** 0.5
    d_jk = (q_jk_x ** 2 + q_jk_y ** 2 + q_jk_z ** 2) ** 0.5
    # Calculate cosine of angle using the dot product formula
    cos = ((q_ji_x * q_jk_x) + (q_ji_y * q_jk_y) + (q_ji_z * q_jk_z)) / (d_ij * d_jk)
    # Raise error if cosine is not withing the range (-1, 1)
    if not (-1 < cos < 1):
        raise ValueError(f"Calculated cosine {cos} does not lie within the range (-1, 1).")
    # Calculate angle from cosine
    angle = math.acos(cos)
    # Calculate potential
    e_ijk = 0.5 * k_a * (angle - angle0) ** 2
    # Calculate common term
    common = k_a * (angle - angle0) / abs(math.sin(angle))
    # Calculate the components of the force vector for 'i'
    f_i_x = common * (q_jk_x / (d_ij * d_jk) - cos * q_ji_x / d_ij ** 2)
    f_i_y = common * (q_jk_y / (d_ij * d_jk) - cos * q_ji_y / d_ij ** 2)
    f_i_z = common * (q_jk_z / (d_ij * d_jk) - cos * q_ji_z / d_ij ** 2)
    # Calculate the components of the force vector for 'k'
    f_k_x = common * (q_ji_x / (d_ij * d_jk) - cos * q_jk_x / d_jk ** 2)
    f_k_y = common * (q_ji_y / (d_ij * d_jk) - cos * q_jk_y / d_jk ** 2)
    f_k_z = common * (q_ji_z / (d_ij * d_jk) - cos * q_jk_z / d_jk ** 2)
    # Calculate the components of the force vector for 'j'
    f_j_x = common * (
        (-q_ji_x - q_jk_x) / (d_ij * d_jk) - cos * (-q_ji_x / d_ij ** 2 - q_jk_x / d_jk ** 2)
    )
    f_j_y = common * (
        (-q_ji_y - q_jk_y) / (d_ij * d_jk) - cos * (-q_ji_y / d_ij ** 2 - q_jk_y / d_jk ** 2)
    )
    f_j_z = common * (
        (-q_ji_z - q_jk_z) / (d_ij * d_jk) - cos * (-q_ji_z / d_ij ** 2 - q_jk_z / d_jk ** 2)
    )
    return f_i_x, f_i_y, f_i_z, f_j_x, f_j_y, f_j_z, f_k_x, f_k_y, f_k_z, e_ijk


def dihedral():
    # TODO (not required for the water model)
    pass


def improper_dihedral():
    # TODO (not required for the water model)
    pass
