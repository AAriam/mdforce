
# Standard library
from typing import Sequence

# 3rd-party packages
import numpy as np

# Self
from ..helpers import check_input_data


class ForceField:
    """
    Force-field of the 3-site flexible SPC water model.
    """

    __slots__ = [
        "q", "mol_ids", "atomic_nums", "bonded_atoms_idx",
        "energy_lj", "energy_coulomb", "energy_bond_vib", "energy_angle_vib",
        "angles", "distances",
        "curr_step"
    ]

    def __init__(
            self,
            q: np.ndarray,
            mol_ids: np.ndarray,
            atomic_nums: np.ndarray,
            bonded_atoms_idx: Sequence[np.ndarray],
            num_steps: int,
    ):
        # Check the input and raise ValueError/TypeError if any discrepancies are found.
        check_input_data(q, mol_ids, atomic_nums, bonded_atoms_idx)

        self.q = q
        self.mol_ids = mol_ids
        self.atomic_nums = atomic_nums
        self.bonded_atoms_idx = bonded_atoms_idx

        num_data_points = num_steps + 1
        self.energy_lj = np.zeros(num_data_points)
        self.energy_coulomb = np.zeros(num_data_points)
        self.energy_bond_vib = np.zeros(num_data_points)
        self.energy_angle_vib = np.zeros(num_data_points)

        num_atoms = mol_ids.size
        self.angles = np.zeros((num_data_points, num_atoms//3))

        self.curr_step = 0

        num_unique_dists = num_atoms * (num_atoms - 1) // 2
        self.distances = np.zeros((num_data_points, num_unique_dists))







