
# Standard library
from typing import Union, Tuple

# 3rd-party packages
import numpy as np


class ForceField:
    """
    Force-field of the 3-site flexible SPC water model.
    This force-field is specifically implemented for input data that exclusively contains water molecules.
    The coordinates vector `q` must be a 2-dimensional numpy array of shape (n, m), where 'n' is the number of
    atoms (must be a multiple of 3), and 'm' is the number of spatial dimensions. Moreover, the coordinates must be
    ordered by the molecules, where the coordinates of the oxygen atom comes first.
    For example, let's say A_n_m denotes the array of coordinates of the mth A atom in nth molecule; then the
    input data must be in the form: [O_1_1, H_1_1, H_1_2, O_2_1, H_2_1, H_2_2, ...]

    Parameters
    ----------
    shape_data : Tuple[int, int]
        Shape of the coordinates vector 'q', where the first value is the number of atoms (a multiple of 3),
        and the second value is the number of spatial dimensions of the coordinates.
    mass_oxygen : Union[float, int, numpy.number]
        Mass of the oxygen atom.
    mass_hydrogen : Union[float, int, numpy.number]
        Mass of the hydrogen atom.
    coulomb_constant : Union[float, int, numpy.number]
        Coulomb constant.
    charge_oxygen : Union[float, int, np.number]
        Electric charge of the oxygen atom.
    charge_hydrogen : Union[float, int, np.number]
        Electric charge of the hydrogen atom.
    lennard_jones_param_a : Union[float, int, np.number]
        Lennard-Jones parameter A for the O–O interaction.
    lennard_jones_param_b : Union[float, int, np.number]
        Lennard-Jones parameter B for the O–O interaction.
    bond_force_constant : Union[float, int, np.number]
        Force constant of the O–H bond vibration.
    bond_eq_dist : Union[float, int, np.number]
        Equilibrium bond length of the O–H bond.
    angle_force_constant : Union[float, int, np.number]
        Force constant of the H–O–H angle vibration.
    angle_eq_angle : Union[float, int, np.number]
        Equilibrium angle of the H–O–H angle.

    Examples
    --------


    """

    __slots__ = [
        "_acceleration", "_force", "_distances", "_distance_vectors", "_angles",
        "_energy_coulomb", "_energy_lj", "_energy_bond", "_energy_angle",
        "_q", "_num_atoms", "_mass_o", "_mass_h", "_coulomb_k", "_charges", "_lj_a", "_lj_b",
        "_bond_k", "_bond_eq_dist", "_angle_k", "_angle_eq"
    ]

    @classmethod
    def from_model(cls, shape_data, model):
        return cls(
            shape_data=shape_data,
            mass_oxygen=model._mass_o_converted.value,
            mass_hydrogen=model._mass_h_converted.value,
            coulomb_constant=model._coulomb_k_converted.value,
            charge_oxygen=model._charge_o_converted.value,
            charge_hydrogen=model._charge_h_converted.value,
            lennard_jones_param_a=model._lj_a.value,
            lennard_jones_param_b=model._lj_b.value,
            bond_force_constant=model._bond_k_oh_converted.value,
            bond_eq_dist=model._bond_eq_len_oh_converted.value,
            angle_force_constant=model._angle_k_hoh_converted.value,
            angle_eq_angle=model._angle_eq_hoh_converted.value
        )

    def __init__(
            self,
            shape_data: Tuple[int, int],
            mass_oxygen: Union[float, int, np.number],
            mass_hydrogen: Union[float, int, np.number],
            coulomb_constant: Union[float, int, np.number],
            charge_oxygen: Union[float, int, np.number],
            charge_hydrogen: Union[float, int, np.number],
            lennard_jones_param_a: Union[float, int, np.number],
            lennard_jones_param_b: Union[float, int, np.number],
            bond_force_constant: Union[float, int, np.number],
            bond_eq_dist: Union[float, int, np.number],
            angle_force_constant: Union[float, int, np.number],
            angle_eq_angle: Union[float, int, np.number]
    ):
        # Store parameters
        self._mass_o = mass_oxygen
        self._mass_h = mass_hydrogen
        self._coulomb_k = coulomb_constant
        self._charges = np.tile([charge_oxygen, charge_hydrogen, charge_hydrogen], shape_data[0]//3)
        self._lj_a = lennard_jones_param_a
        self._lj_b = lennard_jones_param_b
        self._bond_k = bond_force_constant
        self._bond_eq_dist = bond_eq_dist
        self._angle_k = angle_force_constant
        self._angle_eq = angle_eq_angle
        # Initialize attributes for storing the data
        self._num_atoms = shape_data[0]
        self._q = np.zeros(shape_data)
        self._acceleration = np.zeros(shape_data)
        self._force = np.zeros(shape_data)
        self._distances = np.zeros((shape_data[0], shape_data[0]))
        self._distance_vectors = np.zeros((shape_data[0], shape_data[0], shape_data[1]))
        self._angles = np.zeros(shape_data[0] // 3)
        self._energy_coulomb = self._energy_lj = self._energy_bond = self._energy_angle = 0

    @property
    def acceleration(self) -> np.ndarray:
        return self._acceleration

    @property
    def force(self) -> np.ndarray:
        return self._force

    @property
    def energy_total(self) -> float:
        return self._energy_coulomb + self._energy_lj + self._energy_bond + self._energy_angle

    @property
    def energy_coulomb(self) -> float:
        return self._energy_coulomb

    @property
    def energy_lennard_jones(self) -> float:
        return self._energy_lj

    @property
    def energy_bond_vibration(self) -> float:
        return self._energy_bond

    @property
    def energy_angle_vibration(self) -> float:
        return self._energy_angle

    @property
    def distances(self) -> np.ndarray:
        return self._distances

    @property
    def distance_vectors(self) -> np.ndarray:
        return self._distance_vectors

    @property
    def bond_angles(self) -> np.ndarray:
        return self._angles

    def update(self, q: np.ndarray) -> None:
        self.new_state(q)
        self.update_distances()
        self.update_forces_energies()
        self.update_acceleration()
        return

    def new_state(self, q: np.ndarray) -> None:
        self._q[...] = q
        self._force[...] = 0
        self._energy_coulomb = self._energy_lj = self._energy_bond = self._energy_angle = 0
        return

    def update_acceleration(self):
        self._acceleration[::3] = self._force[::3] / self._mass_o
        self._acceleration[1::3] = self._force[1::3] / self._mass_h
        self._acceleration[2::3] = self._force[2::3] / self._mass_h
        return

    def update_forces_energies(self) -> None:
        self.update_coulomb()
        self.update_lennard_jones()
        self.update_bond_vibration()
        self.update_angle_vibration()
        return

    def update_coulomb(self) -> None:

        for idx_curr_atom in range(self._num_atoms - 3):

            idx_first_interacting_atom = idx_curr_atom + 3 - idx_curr_atom % 3
            dists = self._distances[idx_curr_atom, idx_first_interacting_atom:]
            dist_vectors = self._distance_vectors[idx_curr_atom, idx_first_interacting_atom:]

            energy = self._coulomb_k * self._charges[idx_curr_atom] * self._charges[idx_first_interacting_atom:] / dists
            self._energy_coulomb += energy.sum()

            f = (energy / dists ** 2).reshape(-1, 1) * dist_vectors
            self._force[idx_curr_atom] += f.sum(axis=0)
            self._force[idx_first_interacting_atom:] += -f
        return

    def update_lennard_jones(self) -> None:

        for idx_curr_atom in range(0, self._num_atoms - 3, 3):
            idx_first_interacting_atom = idx_curr_atom + 3
            dists = self._distances[idx_curr_atom, idx_first_interacting_atom::3]
            dist_vectors = self._distance_vectors[idx_curr_atom, idx_first_interacting_atom::3]

            inverse_dist_2 = 1 / dists ** 2
            inverse_dist_6 = inverse_dist_2 ** 3

            # Calculate potential
            e_attractive = -self._lj_b * inverse_dist_6
            e_repulsive = self._lj_a * inverse_dist_6 ** 2
            self._energy_lj += (e_repulsive + e_attractive).sum()

            # Calculate force
            f_attractive = 6 * e_attractive
            f_repulsive = 12 * e_repulsive
            f = ((f_attractive + f_repulsive) * inverse_dist_2).reshape(-1, 1) * dist_vectors
            self._force[idx_curr_atom] += f.sum(axis=0)
            self._force[idx_first_interacting_atom::3] += -f
        return

    def update_bond_vibration(self) -> None:

        for idx_curr_atom in range(0, self._num_atoms, 3):
            dists = self._distances[idx_curr_atom, idx_curr_atom + 1:idx_curr_atom + 3]
            dist_vectors = self._distance_vectors[idx_curr_atom, idx_curr_atom + 1:idx_curr_atom + 3]
            displacements = dists - self._bond_eq_dist
            k_times_displacements = self._bond_k * displacements

            self._energy_bond += (k_times_displacements * displacements / 2).sum()

            f = (-k_times_displacements / dists).reshape(-1, 1) * dist_vectors
            self._force[idx_curr_atom] += f.sum(axis=0)
            self._force[idx_curr_atom + 1:idx_curr_atom + 3] += -f
        return

    def update_angle_vibration(self) -> None:

        for idx_curr_atom in range(0, self._num_atoms, 3):
            r_ml = -self._distance_vectors[idx_curr_atom, idx_curr_atom + 1]
            r_mr = -self._distance_vectors[idx_curr_atom, idx_curr_atom + 2]
            dist_ml = self._distances[idx_curr_atom, idx_curr_atom + 1]
            dist_mr = self._distances[idx_curr_atom, idx_curr_atom + 2]
            cos = np.dot(r_ml, r_mr) / (dist_ml * dist_mr)
            angle = np.arccos(cos)
            self._angles[idx_curr_atom//3] = angle

            # Calculate common terms
            sin = np.sin(angle)
            angle_displacement = angle - self._angle_eq
            a = self._angle_k * angle_displacement / sin
            dist_ml_mult_dist_mr = dist_ml * dist_mr
            r_ml_div_dist2_ml = r_ml / dist_ml ** 2
            r_mr_div_dist2_mr = r_mr / dist_mr ** 2

            # Calculate potential
            self._energy_angle += self._angle_k * angle_displacement ** 2

            # Calculate f_l
            b1 = r_mr / dist_ml_mult_dist_mr
            c1 = cos * r_ml_div_dist2_ml
            self._force[idx_curr_atom + 1] += a * (b1 - c1)

            # Calculate f_r
            b2 = r_ml / dist_ml_mult_dist_mr
            c2 = cos * r_mr_div_dist2_mr
            self._force[idx_curr_atom + 2] += a * (b2 - c2)

            # Calculate f_m
            b3 = -(r_ml + r_mr) / dist_ml_mult_dist_mr
            c3 = cos * (r_ml_div_dist2_ml + r_mr_div_dist2_mr)
            self._force[idx_curr_atom] = a * (b3 + c3)
        return

    def update_distances(self) -> None:
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

        for idx, coord in enumerate(self._q[:-1]):
            self._distance_vectors[idx, idx + 1:] = coord - self._q[idx + 1:]
        self._distances[...] = np.linalg.norm(self._distance_vectors, axis=2)
        return
