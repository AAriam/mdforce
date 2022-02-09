

# Standard library
from typing import Union, Tuple

# 3rd-party packages
import numpy as np
from scipy import special

# Self
from .switch_functions import SwitchFunction


class CoulombEwald:
    def __init__(
            self,
            k_e,
            box_lengths,
            switch: SwitchFunction,
    ):
        self._k_e: float = k_e
        self._box_lengths: np.ndarray = box_lengths
        self._switch: SwitchFunction = switch

        self._sigma: float = 0

        self._sqrt2_sigma = np.sqrt(2) * self._sigma

        self._ks: np.ndarray = self._calculate_k_vectors()

        # --- Calculate common terms ---
        # sqrt(2/π)
        self._sqrt_2_div_pi = np.sqrt(2 / np.pi)
        # 1/Vɛ0 (where 1/ɛ0 = 4π*k_e)
        self._inv_v_epsilon = 4 * np.pi * self._k_e / np.prod(self._box_lengths)
        # exp(-σ^2/2)
        self._exp_neg_sigma2_div_2 = np.exp(-self._sigma ** 2 / 2)

        return

    def _calculate_k_vectors(self):
        return

    def __call__(self, q_jsi, d_ijs, c_ijs):
        self._q_jsi = q_jsi
        self._d_ijs = d_ijs
        self._c_ijs = c_ijs
        # Initialize force and energy arrays
        self._f_ijs = np.zeros(q_jsi.shape)
        self._e_ijs = np.zeros(d_ijs.shape)
        # Calculate
        self._calculate_short_range()
        self._calculate_long_range()
        return self._f_ijs, self._e_ijs

    def _calculate_short_range(self):
        # Create mask for distances within the cutoff
        smaller_d_c_mask = self._d_ijs < self._switch.dc
        d_ijs_smaller_d_c = self._d_ijs[smaller_d_c_mask]
        # Calculate common terms
        d_ijs_smaller_d_c__div__sqrt2_sigma = d_ijs_smaller_d_c / self._sqrt2_sigma
        # Calculate full Coulomb for all distances within the cutoff
        # Calculate potentials
        e_ijs = self._k_e * self._c_ijs * special.erfc(
            d_ijs_smaller_d_c__div__sqrt2_sigma
        ) / d_ijs_smaller_d_c
        self._e_ijs[smaller_d_c_mask] = e_ijs
        # Calculate forces
        self._f_ijs[smaller_d_c_mask] = (
                e_ijs + self._k_e * self._c_ijs * self._sqrt_2_div_pi * np.exp(
                    - d_ijs_smaller_d_c__div__sqrt2_sigma ** 2
                )
        ) * self._q_jsi[smaller_d_c_mask] / d_ijs_smaller_d_c ** 2
        # Create mask for distances between d0 and cutoff (within switch region)
        larger_d_0_mask = self._d_ijs > self._switch.d0
        within_switch_mask = np.logical_and(smaller_d_c_mask, larger_d_0_mask)
        # Calculate the switch function's value and its derivative for those distances
        switch_val, switch_deriv_val = self._switch(
            self._q_jsi[within_switch_mask], self._d_ijs[within_switch_mask]
        )
        # Modify forces and energies for distances within the switch
        self._f_ijs[within_switch_mask] = (
                self._f_ijs[within_switch_mask] * switch_val
                - e_ijs[within_switch_mask] * switch_deriv_val
        )
        self._e_ijs[within_switch_mask] *= switch_val
        return

    def _calculate_long_range(self):
        for k in self._ks:

        return


def coulomb(
        q_jsi: np.ndarray,
        d_ijs: np.ndarray,
        c_ijs: Union[np.ndarray, float],
        k_e: float,
) -> Tuple[np.ndarray, np.ndarray]:
    # Calculate potentials
    e_ijs = k_e * c_ijs / d_ijs
    # Calculate forces
    f_ijs = (e_ijs / d_ijs ** 2).reshape(-1, 1) * q_jsi
    return f_ijs, e_ijs


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
        switch: SwitchFunction,
) -> Tuple[np.ndarray, np.ndarray]:
    # Initialize force and energy arrays
    f_ijs = np.zeros(q_jsi.shape)
    e_ijs = np.zeros(d_ijs.shape)
    # Create mask for distances within the cutoff
    smaller_d_c = d_ijs < switch.dc
    # Calculate full Lennard-Jones for all distances within the cutoff
    f_ijs[smaller_d_c], e_ijs[smaller_d_c] = lennard_jones(
        q_jsi[smaller_d_c], d_ijs[smaller_d_c], a_ijs, b_ijs
    )
    # Create mask for distances between d0 and cutoff (within switch region)
    larger_d_0 = d_ijs > switch.d0
    within_switch = np.logical_and(smaller_d_c, larger_d_0)
    # Calculate the switch function's value and its derivative for those distances
    switch_val, switch_deriv_val = switch(q_jsi[within_switch], d_ijs[within_switch])
    # Modify forces and energies for distances within the switch
    f_ijs[within_switch] = (
            f_ijs[within_switch] * switch_val - e_ijs[within_switch] * switch_deriv_val
    )
    e_ijs[within_switch] *= switch_val
    return f_ijs, e_ijs


def bond_vibration_harmonic(
        q_jsi: np.ndarray,
        d_ijs: np.ndarray,
        d0_ijs: float,
        k_b_ijs: float
) -> Tuple[np.ndarray, np.ndarray]:
    # Calculate common terms only once
    delta_d_ijs = d_ijs - d0_ijs
    k__delta_d_ijs = k_b_ijs * delta_d_ijs
    # Calculate potentials
    e_ijs = k__delta_d_ijs * delta_d_ijs / 2
    # Calculate forces on each atom
    f_ijs = (-k__delta_d_ijs / d_ijs).reshape(-1, 1) * q_jsi
    return f_ijs, e_ijs


def angle_vibration_harmonic(
        q_ji: np.ndarray,
        q_jk: np.ndarray,
        d_ij: float,
        d_jk: float,
        angle0: float,
        k_a: float,
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
