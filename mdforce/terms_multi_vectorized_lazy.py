

# Standard library
from typing import Union, Tuple

# 3rd-party packages
import numpy as np


def lennard_jones(
        q_jsi: np.ndarray,
        d_ijs: np.ndarray,
        a_ijs: Union[: np.ndarray, float],
        b_ijs: Union[: np.ndarray, float],
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

