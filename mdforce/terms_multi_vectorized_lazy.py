"""

"""


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
            k_e: float,
            box_lengths: np.ndarray,
            switch: SwitchFunction,
            p: float,
    ):

        # Assign input arguments to instance attributes
        self._k_e: float = k_e
        self._box_lengths: np.ndarray = box_lengths
        self._switch: SwitchFunction = switch
        self._p = p
        # Calculate 'σ' from 'dc' (cutoff in real space) and 'p'
        self._sigma: float = switch.dc / np.sqrt(2*p)
        # Calculate the set of k-vectors to iterate over
        self._calculate_k_vectors()
        # Calculate common terms in calculation of the short-range part
        self._calculate_common_terms_short_range()
        # Calculate common terms in calculation of the long-range part
        self._calculate_common_terms_long_range()
        return

    def _calculate_common_terms_short_range(self):
        # 1 / (sqrt(2) * σ)
        self._inv_sqrt2_sigma: float = 1 / (np.sqrt(2) * self._sigma)
        # sqrt(2/π)
        self._sqrt_2_div_pi: float = np.sqrt(2 / np.pi)
        pass

    def _calculate_common_terms_long_range(self):
        # 1/Vɛ0 (where 1/ɛ0 = 4π*k_e)
        self._inv_v_epsilon: float = 4 * np.pi * self._k_e / np.prod(self._box_lengths)
        # k^2 (i.e. ||k_vector||^2)
        self._k_norms2: np.ndarray = self._k_norms ** 2
        # exp(–σ^2 * k^2 / 2)
        self._exp_neg_sigma2_k2_div_2: np.ndarray = np.exp(-self._sigma ** 2 * self._k_norms2 / 2)
        # exp(–σ^2 * k^2 / 2) / (V * ɛ0 * k^2)
        self._common_energy_long_range: np.ndarray = (
                self._inv_v_epsilon * self._exp_neg_sigma2_k2_div_2 / self._k_norms2
        )
        return

    def _calculate_k_vectors(self):
        # Calculate maximum length of k-vector, from 'p' and 'dc'
        self._k_max = 2 * self._p / self._switch.dc
        # Calculate reciprocal box lengths vector (2π/L_x, 2π/L_y, 2π/L_z) = (k_x, k_y, k_z)
        reciprocal_box_lengths = (2 * np.pi) / self._box_lengths
        # Initialize empty lists to store k-vectors and their norms
        ks = []
        k_norms = []
        # Calculate the maximum value of counter 'n' in each direction (i.e. assuming other
        # counters are zero):
        # k_vector = (n_x * k_x, n_y * k_y, n_z * k_z)
        # ||k_vector|| <= k_max  =>  n_x^2 * k_x^2 + n_y^2 * k_y^2 + n_z^2 * k_z^2 <= k_max^2
        # with n_y = 0, n_z = 0  =>  n_x <= k_max / k_x
        n_x_max = int(self._k_max / reciprocal_box_lengths[0])
        n_y_max = int(self._k_max / reciprocal_box_lengths[1])
        # Calculate the range of each counter
        n_x_range = list(range(-n_x_max, n_x_max + 1))
        n_y_range = list(range(-n_y_max, n_y_max + 1))
        # If the box is 3-dimensional
        if reciprocal_box_lengths.size == 3:
            # Calculate the same for 'n_z'
            n_z_max = int(self._k_max / reciprocal_box_lengths[2])
            n_z_range = list(range(-n_z_max, n_z_max + 1))
            # Iterate over all combinations of 'n_x', 'n_y' and 'n_z'
            for n_x in n_x_range:
                for n_y in n_y_range:
                    for n_z in n_z_range:
                        # Calculate the k-vector and its norm
                        vec = np.array([n_x, n_y, n_z]) * reciprocal_box_lengths
                        norm = np.linalg.norm(vec)
                        # If the norm is smaller/equal k_max, append it to list
                        if norm <= self._k_max:
                            ks.append(vec)
                            k_norms.append(norm)
        # If the box is 2-dimensional
        else:
            # Do the same iteration, but only over 'n_x' and 'n_y'
            for n_x in n_x_range:
                for n_y in n_y_range:
                    vec = np.array([n_x, n_y]) * reciprocal_box_lengths
                    norm = np.linalg.norm(vec)
                    if norm <= self._k_max:
                        ks.append(vec)
                        k_norms.append(norm)
        # Turn lists into arrays
        ks = np.array(ks)
        k_norms = np.array(k_norms)
        # Find the index of the array (0,0,0), and delete it from k-vectors and norms
        idx_zero = np.where((k_norms == 0))[0][0]
        self._k_vectors = np.delete(ks, idx_zero, axis=0)
        self._k_norms = np.delete(k_norms, idx_zero)
        return

    def __call__(self, q_jsi, d_ijs, c_ijs):
        self._q_jsi = q_jsi
        self._d_ijs = d_ijs
        self._c_ijs = c_ijs
        # Initialize force and energy arrays
        self._f_ijs = np.zeros(q_jsi.shape)
        self._e_ijs = np.zeros(d_ijs.shape)

        self._f_ijs_long = np.zeros(q_jsi.shape)
        self._e_ijs_long = np.zeros(d_ijs.shape)

        # Calculate
        self._calculate_short_range()
        self._calculate_long_range()

        if test:
            e_long = self._calculate_long_range_script(q, cs)
            e_self2 = (cs ** 2).sum() * self._common_energy_long_range.sum() / 2
            print(e_long, "-", e_self2 + self._e_ijs_long.sum())

        return self._f_ijs, self._e_ijs

    def _calculate_short_range(self):
        # Create mask for distances within the cutoff
        smaller_dc_mask = self._d_ijs < self._switch.dc
        # Apply mask to distances and charges
        d_ijs_smaller_dc = self._d_ijs[smaller_dc_mask]
        c_ijs_smaller_dc = self._c_ijs[smaller_dc_mask]
        # Calculate common terms
        d_ijs_smaller_dc__div__sqrt2_sigma = d_ijs_smaller_dc * self._inv_sqrt2_sigma
        k_e__c_ijs_smaller_dc = self._k_e * c_ijs_smaller_dc
        # Calculate full short-range Coulomb for all distances within the cutoff
        # Calculate potentials
        e_ijs = k_e__c_ijs_smaller_dc * special.erfc(
            d_ijs_smaller_dc__div__sqrt2_sigma
        ) / d_ijs_smaller_dc
        self._e_ijs[smaller_dc_mask] = e_ijs
        # Calculate forces
        self._f_ijs[smaller_dc_mask] = (
            (
                e_ijs + k_e__c_ijs_smaller_dc * self._sqrt_2_div_pi * np.exp(
                    - d_ijs_smaller_dc__div__sqrt2_sigma ** 2
                ) / self._sigma
            ) / d_ijs_smaller_dc ** 2
        ).reshape(-1, 1) * self._q_jsi[smaller_dc_mask]
        # Create mask for distances between d0 and cutoff (within switch region)
        larger_d0_mask = self._d_ijs > self._switch.d0
        within_switch_mask = np.logical_and(smaller_dc_mask, larger_d0_mask)
        # Calculate the switch function's value and its derivative for those distances
        switch_val, switch_deriv_val = self._switch(
            self._q_jsi[within_switch_mask], self._d_ijs[within_switch_mask]
        )
        # Modify forces and energies for distances within the switch
        self._f_ijs[within_switch_mask] = (
                self._f_ijs[within_switch_mask] * switch_val.reshape(-1, 1)
                - self._e_ijs[within_switch_mask].reshape(-1, 1) * switch_deriv_val
        )
        self._e_ijs[within_switch_mask] *= switch_val
        return

    def _calculate_long_range(self):

        # Calculate common terms for energy, i.e.
        # c_ijs * exp(–σ^2 * k^2 / 2) / (V * ɛ0 * k^2)
        common_terms_energy_long_range = self._common_energy_long_range.reshape(
            -1, 1
        ) * self._c_ijs
        # Calculate common terms for force, i.e.
        # k_vectors * c_ijs * exp(–σ^2 * k^2 / 2) / (V * ɛ0 * k^2)
        common_terms_force_long_range = common_terms_energy_long_range.reshape(
            (common_terms_energy_long_range.shape[0], -1, 1)
        ) * self._k_vectors.reshape((-1, 1, self._k_vectors.shape[1]))
        # Iterate over k vectors and add the calculated forces and potential to the main arrays
        for idx, k in enumerate(self._k_vectors):
            k_dot_q_jsi = np.dot(self._q_jsi, k)
            cos = np.cos(k_dot_q_jsi)
            sin = np.sin(k_dot_q_jsi)
            self._e_ijs_long += common_terms_energy_long_range[idx] * cos
            self._f_ijs_long += common_terms_force_long_range[idx] * sin.reshape(-1, 1)
        return

    def _calculate_long_range_script(self, q, cs):
        e = 0
        for idx, k in enumerate(self._k_vectors):
            s = 0 + 0j
            for i in range(cs.size):
                s += cs[i] * np.exp(1j * np.dot(k, q[i]))
            e += s * np.conj(s) * self._common_energy_long_range[idx] / 2
        return e


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
            f_ijs[within_switch] * switch_val.reshape(-1, 1)
            - e_ijs[within_switch].reshape(-1, 1) * switch_deriv_val
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
