
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
        "force",
        "energy_lj", "energy_coulomb", "energy_bond_vib", "energy_angle_vib",
        "angles", "distance_vectors", "distances", "num_atoms",
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

        self.num_atoms = mol_ids.size
        self.curr_step = 0

        sort_mask = self._create_sorting_mask(mol_ids, atomic_nums)
        self.q = q[sort_mask]
        self.mol_ids = mol_ids[sort_mask]
        self.atomic_nums = atomic_nums[sort_mask]
        # self.bonded_atoms_idx = bonded_atoms_idx[sort_mask]

        self.force = np.zeros_like(self.q)

        num_data_points = num_steps + 1
        self.energy_lj = np.zeros(num_data_points)
        self.energy_coulomb = np.zeros(num_data_points)
        self.energy_bond_vib = np.zeros(num_data_points)
        self.energy_angle_vib = np.zeros(num_data_points)

        self.angles = np.zeros((num_data_points, self.num_atoms//3))

        num_unique_dists = self.num_atoms * (self.num_atoms - 1) // 2
        self.distance_vectors = np.zeros((num_data_points, num_unique_dists, self.q.shape[1]))
        self.distances = np.zeros((num_data_points, num_unique_dists))

    def calculate_distances(self) -> None:
        """
        Calculate the distance vector and distance between all unique pairs of atoms at the current step.

        Returns
        -------
            None
            Distance vectors and distances are stored in `self.distance_vectors` and `self.distances` respectively,
            at the position defined by `self.curr_step`.

        Notes
        -----
        The distances 'q_i - q_j` are calculated for all unique pairs, where 'i' is smaller than 'j'.
        Distances 'q_i - q_j` where 'i' is larger than 'j' are equal to the negative value of 'q_j - q_i'.
        """
        start = 0
        end = self.num_atoms - 1
        for idx, coord in enumerate(self.q):
            self.distance_vectors[self.curr_step, start:end] = coord - self.q[idx + 1:]
            start = end
            end += (self.num_atoms - idx - 2)
        self.distances[self.curr_step] = np.linalg.norm(self.distance_vectors[self.curr_step], axis=1)
        return

    def dist_idx(self, i, j):
        """
        Calculate the index of the element in `self.distances` or `self.distance_vectors` corresponding to the
        distance 'q_i - q_j', where 'i' and 'j' are the indices of the two atoms in `self.q`.

        Parameters
        ----------
        i : int
            Index of particle 'q_i' in `self.q`.
        j : int
            Index of particle `q_j` in `self.q`.

        Returns
        -------
        idx : int
            Index of the element in `self.distances` or `self.distance_vectors` corresponding to 'q_i - q_j'.

        Notes
        -----
        `i` should always be smaller than `j`, otherwise the returned index will be incorrect.
        To get the distances 'q_i - q_j` where 'i' is larger than 'j', ask for the distance 'q_j - q_i',
        and then multiply the distance with -1.
        """
        return int(j - 1 - i * (i / 2 + 1.5 - self.num_atoms))

    def dist_idx_first(self, atom_idx):
        """
        Calculate the index of first element in `self.distances` or `self.distance_vectors` that corresponds
        to the distance between a target atom and the rest of atoms with larger indices.

        Parameters
        ----------
        atom_idx : int
            Index of the target atom in `self.q`.

        Returns
        -------
            int
        """
        return int(atom_idx * (self.num_atoms - (atom_idx + 1) / 2))



    def dist_idx_coulomb(self, idx_target_atom, idx_first_interacting_atom):
        stop = (idx_target_atom + 1) * int(self.num_atoms - 1 - idx_target_atom/2)
        start = stop - self.num_atoms + idx_first_interacting_atom
        return start, stop

    def coulomb(self):
        idx_dist_first = 2
        idx_dist_last = self.num_atoms - 1

        for idx_curr_atom, coord in enumerate(self.q):
            idx_first_interacting_atom = idx_curr_atom + 3 - idx_curr_atom % 3



            e = self.k_coulomb * self.charges[idx_curr_atom] * self.charges[idx_first_interacting_atom:]
            f = (e / self.distances[idx_dist_first:idx_dist_last] ** 2).reshape(-1, 1) * (
                    self.distance_vectors[idx_dist_first:idx_dist_last]
            )
            self.force[idx_first_interacting_atom:] += -f
            self.force[idx_curr_atom] += f.sum(axis=0)
            idx_dist_first = idx_dist_last + (1 - idx_curr_atom) % 3
            idx_dist_last += (self.num_atoms - idx_curr_atom - 2)

    def lennard_jones(self):

        for idx_curr_atom in range(2, self.num_atoms, 3):
            dists = self.distances


    @staticmethod
    def _create_sorting_mask(
            mol_ids: np.ndarray,
            atomic_nums: np.ndarray
    ) -> np.ndarray:
        """
        Create an array of atom indices, where the atoms are first sorted
        by their molecule-ID, and then by their atom type.

        Parameters
        ----------
        mol_ids : numpy.ndarray
            1D-array of shape (n, ), containing the ID of the molecule, to which each atom in `q` belongs to.
        atomic_nums : numpy.ndarray
            1D array of shape (n, ), containing the atomic number of each atom in `q`.

        Returns
        -------
            numpy.ndarray
            Index array of all atoms in the input data, first sorted by their molecule-ID, and then by their atom type.

        Examples
        --------
        Let's say A_n_m denotes the nth A atom in mth molecule;
        then an input data `q` = [H_1_1, H_1_2, H_2_1, H_2_2, O_1_1, O_1_2],
        will have `atom_types` = [1, 1, 1, 1, 8, 8]
        and `mol_ids` = [1, 2, 1, 2, 1, 2].
        Applying this function to `atom_types` and `mol_ids` will then return:
        [0, 2, 4, 1, 3, 5]
        Therefore, applying this index array to `q` will return:
        [H_1_1, H_2_1, O_1_1, H_1_2, H_2_2, O_1_2]
        """
        # Create array of atom indices
        atom_idx = np.arange(atomic_nums.size)
        # Calculate new indices when atoms are sorted by their molecule ID
        atom_idx_sorted_by_mol_id = mol_ids.argsort()
        # Update atom indices
        atom_idx = atom_idx[atom_idx_sorted_by_mol_id]
        # Get array of atom types based on sorting
        atom_types_sorted = atomic_nums[atom_idx_sorted_by_mol_id]
        # Calculate new indices when atoms are again sorted, now by their atom type
        for n in range(0, atom_types_sorted.shape[0] - 2, 3):
            atom_types_sorted_idx = atom_types_sorted[n:n + 3].argsort() + n
            atom_idx[n:n + 3][...] = atom_idx[atom_types_sorted_idx]
        return atom_idx




