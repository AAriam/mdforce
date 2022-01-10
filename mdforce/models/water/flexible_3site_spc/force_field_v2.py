
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
        "k_coulomb", "charges",
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

        self.distance_vectors = np.zeros((num_data_points, self.num_atoms, self.num_atoms, self.q.shape[1]))
        self.distances = np.zeros((num_data_points, self.num_atoms, self.num_atoms))

    @property
    def force(self):
        return self._force

    @property
    def energy_total(self):
        return self._e_coulomb + self._e_lj + self._e_bond + self._e_angle

    @property
    def energy_coulomb(self):
        return self._e_coulomb

    @property
    def energy_lennard_jones(self):
        return self._e_lj

    @property
    def energy_bond_vibration(self):
        return self._e_bond

    @property
    def energy_angle_vibration(self):
        return self._e_angle

    @property
    def distances(self):
        return self._distances

    @property
    def distance_vectors(self):
        return self._distant_vectors

    @property
    def bond_angles(self):
        return self._angles

    def update(self, q):
        self._q[...] = q
        self.coulomb()
        self.lennard_jones()
        self.bond_vibration_harmonic()
        self.angle_vibration_harmonic()
        return

    def coulomb(self):

        for idx_curr_atom, coord in enumerate(self.q[:-3]):

            idx_first_interacting_atom = idx_curr_atom + 3 - idx_curr_atom % 3
            dists = self.distances[self.curr_step, idx_curr_atom, idx_first_interacting_atom:]
            dist_vectors = self.distance_vectors[self.curr_step, idx_curr_atom, idx_first_interacting_atom:]

            e = self.k_coulomb * self.charges[idx_curr_atom] * self.charges[idx_first_interacting_atom:] / dists
            self.energy_coulomb[self.curr_step] += e

            f = (e / dists ** 2).reshape(-1, 1) * dist_vectors
            self.force[idx_curr_atom] += f.sum(axis=0)
            self.force[idx_first_interacting_atom:] += -f
        return

    def lennard_jones(self):

        for idx_curr_atom in range(0, self.num_atoms - 3, 3):
            idx_first_interacting_atom = idx_curr_atom + 3
            dists = self.distances[self.curr_step, idx_curr_atom, idx_first_interacting_atom::3]
            dist_vectors = self.distance_vectors[self.curr_step, idx_curr_atom, idx_first_interacting_atom::3]

            inverse_dist_2 = 1 / dists ** 2
            inverse_dist_6 = inverse_dist_2 ** 3

            # Calculate potential
            e_attractive = -self.params["lj_b"] * inverse_dist_6
            e_repulsive = self.params["lj_a"] * inverse_dist_6 ** 2
            self.energy_lj[self.curr_step] += (e_repulsive + e_attractive).sum()

            # Calculate force
            f_attractive = 6 * e_attractive
            f_repulsive = 12 * e_repulsive
            f = ((f_attractive + f_repulsive) * inverse_dist_2).reshape(-1, 1) * dist_vectors
            self.force[idx_curr_atom] += f.sum(axis=0)
            self.force[idx_first_interacting_atom::3] += -f
        return

    def bond_vibration_harmonic(self):

        for idx_curr_atom in range(0, self.num_atoms, 3):
            dists = self.distances[self.curr_step, idx_curr_atom, idx_curr_atom + 1:idx_curr_atom + 3]
            dist_vectors = self.distance_vectors[self.curr_step, idx_curr_atom, idx_curr_atom + 1:idx_curr_atom + 3]
            displacements = dists - self.params["bond_eq_dist"]
            k_times_displacements = self.params["bond_vib_k"]

            e = (k_times_displacements * displacements / 2).sum()
            self.energy_bond_vib[self.curr_step] += e

            f = (-k_times_displacements / dists).reshape(-1, 1) * dist_vectors
            self.force[idx_curr_atom] += f.sum(axis=0)
            self.force[idx_curr_atom + 1:idx_curr_atom + 3] += -f
        return

    def angle_vibration_harmonic(self):

        for idx_curr_atom in range(0, self.num_atoms, 3):
            r_ml = -self.distance_vectors[self.curr_step, idx_curr_atom, idx_curr_atom + 1]
            r_mr = -self.distance_vectors[self.curr_step, idx_curr_atom, idx_curr_atom + 2]
            dist_ml = self.distances[self.curr_step, idx_curr_atom, idx_curr_atom + 1]
            dist_mr = self.distances[self.curr_step, idx_curr_atom, idx_curr_atom + 2]
            cos = np.dot(r_ml, r_mr) / (dist_ml * dist_mr)
            angle = np.arccos(cos)
            self.angles[self.curr_step, idx_curr_atom//3] = angle

            # Calculate common terms
            sin = np.sin(angle)
            angle_displacement = angle - self.params["angle_eq"]
            a = self.params["angle_vib_k"] * angle_displacement / sin
            dist_ml_mult_dist_mr = dist_ml * dist_mr
            r_ml_div_dist2_ml = r_ml / dist_ml ** 2
            r_mr_div_dist2_mr = r_mr / dist_mr ** 2

            # Calculate potential
            self.energy_angle_vib[self.curr_step] += self.params["angle_vib_k"] * angle_displacement ** 2

            # Calculate f_l
            b1 = r_mr / dist_ml_mult_dist_mr
            c1 = cos * r_ml_div_dist2_ml
            self.force[idx_curr_atom + 1] += a * (b1 - c1)

            # Calculate f_r
            b2 = r_ml / dist_ml_mult_dist_mr
            c2 = cos * r_mr_div_dist2_mr
            self.force[idx_curr_atom + 2] += a * (b2 - c2)

            # Calculate f_m
            b3 = (
                2 * self.q[idx_curr_atom] - self.q[idx_curr_atom + 1] - self.q[idx_curr_atom + 2]
            ) / dist_ml_mult_dist_mr
            c3 = cos * (r_ml_div_dist2_ml + r_mr_div_dist2_mr)
            self.force[idx_curr_atom] = a * (b3 + c3)
        return

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

        for idx, coord in enumerate(self.q[:-1]):
            self.distance_vectors[self.curr_step, idx, idx + 1:] = coord - self.q[idx + 1:]

        self.distances[self.curr_step] = np.linalg.norm(self.distance_vectors[self.curr_step], axis=2)
        return

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
            atom_idx[n:n + 3][...] = np.flip(atom_idx[atom_types_sorted_idx])
        return atom_idx
