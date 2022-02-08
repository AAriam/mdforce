"""
Module containing classes representing switch functions for truncating short-range interactions.
"""

# Standard library
from typing import Tuple

# 3rd-party packages
import numpy as np


class SwitchFunction:
    """
    Superclass for all switch function classes.
    """
    def __init__(self, distance_start: float, distance_end: float):
        self._d0 = distance_start
        self._dc = distance_end
        # Calculate recurring terms to use in each call
        return

    def __call__(self, q_jsi: np.ndarray, d_ijs: np.ndarray):
        pass

    @property
    def d0(self):
        return self._d0

    @property
    def dc(self):
        return self._dc


class Poly1(SwitchFunction):
    """
    Polynomial switch function S(r) with the formula
    S(r) = 1 + [(r^2 - a^2) / (b^2 - a^2)]^2 [2((r^2 - a^2) / (b^2 - a^2)) - 3]
    where 'r' is the distance, 'a' is the distance at which the switch starts (i.e. the distance at
    which the function's value is 1), and 'b' is the cutoff distance (i.e. the distance at which
    the function's value becomes 0).

    """
    def __init__(self, distance_start: float, distance_end: float):
        super().__init__(distance_start, distance_end)
        # Calculate recurring terms to use in each call
        self._d0_2 = self._d0 ** 2
        self._dc_2 = self._dc ** 2
        self._dc_2__d0_2 = self._dc_2 - self._d0_2
        self._inv3_dc_2__d0_2 = 1 / (self._dc_2__d0_2 ** 3)
        self._12_inv3_dc_2__d0_2 = 12 * self._inv3_dc_2__d0_2
        return

    def __call__(self, q_jsi: np.ndarray, d_ijs: np.ndarray):
        d_2 = d_ijs ** 2
        d_2__d0_2 = d_2 - self._d0_2
        d_2__dc_2 = d_2 - self._dc_2
        s = self._inv3_dc_2__d0_2 * d_2__dc_2 ** 2 * (2 * d_2__d0_2 + self._dc_2__d0_2)
        ds = (self._12_inv3_dc_2__d0_2 * d_2__d0_2 * d_2__dc_2).reshape(-1, 1) * q_jsi
        return s, ds

    def _switch_test(self, q_jsi, d_ijs):
        """
        Calculate the value of the switch function and its derivative, using a non-simplified and
        non-optimized implementation, for testing purposes.

        Parameters
        ----------
        q_jsi : numpy.ndarray
        d_ijs : numpy.ndarray

        Returns
        -------
        s, ds : Tuple[numpy.ndarray, numpy.ndarray]
        """
        # Calculate the value of switch function
        s = 1 + (
            ((d_ijs ** 2 - self._d0_2) / (self._dc_2 - self._d0_2)) ** 2
            * ((2 * (d_ijs ** 2 - self._d0_2) / (self._dc_2 - self._d0_2)) - 3)
        )
        # Calculate the value of its derivative
        ds = (
            (
                2
                * ((d_ijs ** 2 - self._d0_2) / (self._dc_2 - self._d0_2))
                * (2 * d_ijs / (self._dc_2 - self._d0_2))
                * ((2 * (d_ijs ** 2 - self._d0_2) / (self._dc_2 - self._d0_2)) - 3)
                + ((d_ijs ** 2 - self._d0_2) / (self._dc_2 - self._d0_2)) ** 2
                * (4 * d_ijs / (self._dc_2 - self._d0_2))
            ).reshape(-1, 1)
            * q_jsi
            / d_ijs.reshape(-1, 1)
        )
        return s, ds
