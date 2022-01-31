
# Standard library
from typing import Union, Tuple

# 3rd-party packages
import numpy as np


def lennard_jones_switch_1(d_ijs, d0_2, dc_2__d0_2):
    """
    S(d_ij) = 1 + ((d_ij^2 - d_0^2) / (d_c^2 - d_0^2))^2 *
             (2 * ((d_ij^2 - d_0^2) / (d_c^2 - d_0^2)) - 3)

    Parameters
    ----------
    d_ijs
    d_0
    d_c

    Returns
    -------

    """
    # Calculate common term
    c = d_ijs ** 2 - d0_2 / dc_2__d0_2
    # Calculate function's value
    s = 1 + c ** 2 * (2 * c - 3)
    # Calculate negative derivative of function
    ds = None
    return s, ds