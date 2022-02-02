

# Standard library
from typing import Union, Tuple

# 3rd-party packages
import numpy as np

# Self
from . import switch_functions as switch


def coulomb():
    pass


def lennard_jones(
        q_jsi: np.ndarray,
        d_ijs: np.ndarray,
        a_ijs: Union[np.ndarray, float],
        b_ijs: Union[np.ndarray, float],
) -> Tuple[np.ndarray, np.ndarray]:

    # Calculate common terms
    inv_d2 = 1 / d_ijs ** 2
    inv_d6 = inv_d2 ** 3
    # Calculate potentials
    e_ijs_repulsive = a_ijs * inv_d6 ** 2
    e_ijs_attractive = -b_ijs * inv_d6
    e_ijs = e_ijs_repulsive + e_ijs_attractive
    # Calculate forces
    f_ijs = (6 * (e_ijs + e_ijs_repulsive) * inv_d2).reshape(-1, 1) * q_jsi
    return f_ijs, e_ijs


def lennard_jones_switch(
        q_jsi: np.ndarray,
        d_ijs: np.ndarray,
        a_ijs: Union[np.ndarray, float],
        b_ijs: Union[np.ndarray, float],
        d_switch_start: float,
        d_switch_end: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    smaller_d_c = d_ijs < d_switch_end
    f_ijs, e_ijs = lennard_jones(
        q_jsi[smaller_d_c], d_ijs[smaller_d_c], a_ijs, b_ijs
    )
    larger_equal_d_0 = d_ijs >= d_switch_start
    within_switch = np.logical_and(smaller_d_c, larger_equal_d_0)
    switch_val, switch_neg_deriv_val = switch.lennard_jones_1(
        d_ijs[within_switch], self.__lj_d02, self.__lj_dc2_d02
    )
    f_ijs[within_switch] = (
            f_ijs[within_switch] * switch_val + e_ijs[within_switch] * switch_neg_deriv_val
    )
    e_ijs[within_switch] *= switch_val
    return f_ijs, e_ijs, smaller_d_c


def bond_vibration_harmonic(
        q_jsi: np.ndarray,
        d_ijs: np.ndarray,
        d0: float,
        k_b: float
) -> Tuple[np.ndarray, np.ndarray]:
    # Calculate common terms only once
    delta_d_ijs = d_ijs - d0
    k__delta_d_ijs = k_b * delta_d_ijs
    # Calculate the potential of the whole molecule
    e_ijs = (k__delta_d_ijs * delta_d_ijs / 2).sum()
    # Calculate forces on each atom
    f_ijs = (-k__delta_d_ijs / d_ijs).reshape(-1, 1) * q_jsi
    return f_ijs, e_ijs


def angle_vibration_harmonic(
        q_ji,
        q_jk,
        d_ij,
        d_jk,
        angle0,
        k_a,
):
    # Calculate common term
    d_ij__d_jk = d_ij * d_jk
    # Calculate the angle from the dot product formula
    cos = np.dot(q_ji, q_jk) / d_ij__d_jk
    angle = np.arccos(cos)
    # Calculate common terms
    delta_angle = angle - angle0
    a = k_a * delta_angle / abs(np.sin(angle))
    # Calculate the potential
    e_ijk = 0.5 * k_a * delta_angle ** 2
    # Calculate force on first hydrogen
    f_i = a * (q_jk / d_ij__d_jk - cos * q_ji / d_ij ** 2)
    # Calculate force on second hydrogen
    f_k = a * (q_ji / d_ij__d_jk - cos * q_jk / d_jk ** 2)
    # Calculate the force on oxygen
    f_j = -(f_i + f_k)
    return f_i, f_j, f_k, e_ijk, angle
