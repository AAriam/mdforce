# Standard library
from typing import Union, Tuple
from pathlib import Path
import webbrowser

# 3rd-party packages
import numpy as np
import pandas as pd
import duq

# Self
from mdforce.data.element_masses import masses


__all__ = ["ForceField"]


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
    """

    __slots__ = [
        "_acceleration",
        "_force",
        "_distances",
        "_distance_vectors",
        "_angles",
        "_energy_coulomb",
        "_energy_lj",
        "_energy_bond",
        "_energy_angle",
        "_q",
        "_num_atoms",
        "_num_molecules",
        "_mass_o",
        "_mass_h",
        "_coulomb_k",
        "_coulomb_k_converted",
        "_charge_o",
        "_charge_o_converted",
        "_charge_h",
        "_charge_h_converted",
        "_charges",
        "_lj_epsilon_oo",
        "_lj_epsilon_oo_converted",
        "_lj_sigma_oo",
        "_lj_sigma_oo_converted",
        "_lj_a",
        "_lj_b",
        "_bond_k",
        "_bond_k_converted",
        "_bond_eq_len",
        "_bond_eq_len_converted",
        "_angle_k",
        "_angle_k_converted",
        "_angle_eq",
        "_angle_eq_converted",
        "_model_name",
        "_model_description",
        "_model_ref_name",
        "_model_ref_cite",
        "_model_ref_link",
        "_desc_charge_o",
        "_desc_charge_h",
        "_desc_lj_epsilon_oo",
        "_desc_lj_sigma_oo",
        "_desc_bond_k",
        "_desc_bond_eq_len",
        "_desc_angle_k",
        "_desc_angle_eq",
        "_unit_mass",
        "_unit_length",
        "_unit_time",
        "_unit_charge",
        "_unit_force",
        "_unit_energy",
        "_fitted",
        "__c_o",
        "__c_h",
        "__k_e",
        "__lj_a",
        "__lj_b",
        "__k_b",
        "__eq_dist",
        "__k_a",
        "__eq_angle",
    ]

    _dataframe = pd.read_pickle(
        Path(__file__).parent.parent.parent / "data/model_params/water_flexible_3site_spc.pkl"
    )

    @classmethod
    def print_available_models(cls):
        for name in cls._dataframe.index[1:]:
            model = cls.from_model(name)
            print(model.model_metadata + "\n\n" + model.model_parameters + "\n")
        return

    @classmethod
    def from_model(cls, model_name):

        charge_o = cls._dataframe.loc[model_name, ("Parameters", "Coulomb", "q_O")].split()
        charge_h = cls._dataframe.loc[model_name, ("Parameters", "Coulomb", "q_H")].split()
        lj_epsilon_oo = cls._dataframe.loc[
            model_name, ("Parameters", "Lennard-Jones", "ε_OO")
        ].split()
        lj_sigma_oo = cls._dataframe.loc[
            model_name, ("Parameters", "Lennard-Jones", "σ_OO")
        ].split()
        bond_k = cls._dataframe.loc[model_name, ("Parameters", "Bond vibration", "k")].split()
        bond_eq_len = cls._dataframe.loc[
            model_name, ("Parameters", "Bond vibration", "r_OH")
        ].split()
        angle_k = cls._dataframe.loc[model_name, ("Parameters", "Angle vibration", "k")].split()
        angle_eq = cls._dataframe.loc[
            model_name, ("Parameters", "Angle vibration", "θ_HOH")
        ].split()

        forcefield = cls(
            charge_oxygen=duq.Quantity(float(charge_o[0]), charge_o[1]),
            charge_hydrogen=duq.Quantity(float(charge_h[0]), charge_h[1]),
            lennard_jones_epsilon_oo=duq.Quantity(float(lj_epsilon_oo[0]), lj_epsilon_oo[1]),
            lennard_jones_sigma_oo=duq.Quantity(float(lj_sigma_oo[0]), lj_sigma_oo[1]),
            bond_force_constant=duq.Quantity(float(bond_k[0]), bond_k[1]),
            bond_eq_dist=duq.Quantity(float(bond_eq_len[0]), bond_eq_len[1]),
            angle_force_constant=duq.Quantity(float(angle_k[0]), angle_k[1]),
            angle_eq_angle=duq.Quantity(float(angle_eq[0]), angle_eq[1]),
        )

        forcefield._model_name = model_name

        forcefield._model_description = cls._dataframe.loc[
            model_name, ("Metadata", "Info", "Description")
        ]
        forcefield._model_ref_name = cls._dataframe.loc[
            model_name, ("Metadata", "Reference", "Name")
        ]
        forcefield._model_ref_cite = cls._dataframe.loc[
            model_name, ("Metadata", "Reference", "Citation")
        ]
        forcefield._model_ref_link = cls._dataframe.loc[
            model_name, ("Metadata", "Reference", "Link")
        ]
        return forcefield

    def __init__(
        self,
        charge_oxygen: Union[str, duq.Quantity],
        charge_hydrogen: Union[str, duq.Quantity],
        lennard_jones_epsilon_oo: Union[str, duq.Quantity],
        lennard_jones_sigma_oo: Union[str, duq.Quantity],
        bond_force_constant: Union[str, duq.Quantity],
        bond_eq_dist: Union[str, duq.Quantity],
        angle_force_constant: Union[str, duq.Quantity],
        angle_eq_angle: Union[str, duq.Quantity],
    ):

        # Store parameters
        self._charge_o = charge_oxygen
        self._charge_h = charge_hydrogen
        self._lj_epsilon_oo = lennard_jones_epsilon_oo
        self._lj_sigma_oo = lennard_jones_sigma_oo
        self._bond_k = bond_force_constant
        self._bond_eq_len = bond_eq_dist
        self._angle_k = angle_force_constant
        self._angle_eq = angle_eq_angle

        # Load constants
        self._mass_o = masses[8]  # duq.Quantity(masses[8], "Da")
        self._mass_h = masses[1]  # duq.Quantity(masses[1], "Da")
        self._coulomb_k = duq.predefined_constants.coulomb_const

        # Attributes that are only set when instantiating from alt. constructor `from_model`
        self._model_name = None
        self._model_description = None
        self._model_ref_name = None
        self._model_ref_cite = None
        self._model_ref_link = None

        # Read parameters description from dataframe
        self._desc_charge_o = self._dataframe.loc["Description", ("Parameters", "Coulomb", "q_O")]
        self._desc_charge_h = self._dataframe.loc["Description", ("Parameters", "Coulomb", "q_H")]
        self._desc_lj_epsilon_oo = self._dataframe.loc[
            "Description", ("Parameters", "Lennard-Jones", "ε_OO")
        ]
        self._desc_lj_sigma_oo = self._dataframe.loc[
            "Description", ("Parameters", "Lennard-Jones", "σ_OO")
        ]
        self._desc_bond_k = self._dataframe.loc[
            "Description", ("Parameters", "Bond vibration", "k")
        ]
        self._desc_bond_eq_len = self._dataframe.loc[
            "Description", ("Parameters", "Bond vibration", "r_OH")
        ]
        self._desc_angle_k = self._dataframe.loc[
            "Description", ("Parameters", "Angle vibration", "k")
        ]
        self._desc_angle_eq = self._dataframe.loc[
            "Description", ("Parameters", "Angle vibration", "θ_HOH")
        ]

        # Attributes that are set after calling `fit_to_input_data`
        self._coulomb_k_converted = None
        self._charge_o_converted = None
        self._charge_h_converted = None
        self._lj_epsilon_oo_converted = None
        self._lj_sigma_oo_converted = None
        self._lj_b = None
        self._lj_a = None
        self._bond_k_converted = None
        self._angle_eq_converted = None
        self._angle_k_converted = None
        self._bond_eq_len_converted = None
        self._unit_charge = None
        self._unit_mass = None
        self._unit_length = None
        self._unit_time = None
        self._unit_force = None
        self._unit_energy = None
        self._fitted = False  # Whether the model has been fitted to input data.

        # Initialize attributes for storing the data
        self._num_molecules = None
        self._num_atoms = None
        self._charges = None
        self._q = None
        self._acceleration = None
        self._force = None
        self._distances = None
        self._distance_vectors = None
        self._angles = None
        self._energy_coulomb = 0
        self._energy_lj = 0
        self._energy_bond = 0
        self._energy_angle = 0

    def fit_to_input_data(
        self, num_molecules, num_atoms, num_spatial_dimensions, unit_length, unit_time
    ):
        """

        Parameters
        ----------
        shape_data : Tuple[int, int]
        Shape of the coordinates vector 'q', where the first value is the number of atoms (a multiple of 3),
        and the second value is the number of spatial dimensions of the coordinates.

        num_molecules
        num_atoms
        num_spatial_dimensions

        Returns
        -------

        """

        self._fit_units(unit_length, unit_time)

        self._num_atoms = num_atoms
        self._num_molecules = num_molecules
        shape_data = (num_atoms, num_spatial_dimensions)
        self._q = np.zeros(shape_data)
        self._acceleration = np.zeros(shape_data)
        self._force = np.zeros(shape_data)
        self._distances = np.zeros((num_atoms, num_atoms))
        self._distance_vectors = np.zeros((num_atoms, num_atoms, num_spatial_dimensions))
        self._angles = np.zeros(num_molecules)
        self._charges = np.tile(
            [
                self._charge_o_converted.value,
                self._charge_h_converted.value,
                self._charge_h_converted.value,
            ],
            num_molecules,
        )
        return

    def _fit_units(self, unit_length, unit_time):

        if isinstance(unit_length, str):
            self._unit_length = duq.Unit(unit_length)
        else:
            self._unit_length = unit_length
        if isinstance(unit_time, str):
            self._unit_time = duq.Unit(unit_time)
        else:
            self._unit_time = unit_time

        self._unit_charge = duq.Unit("e")
        self._unit_mass = duq.Unit("Da")

        self._unit_force = self._unit_mass * self._unit_length / self._unit_time ** 2
        self._unit_energy = self._unit_force * self._unit_length

        # Coulomb
        self._charge_o_converted = self._charge_o.convert_unit(self._unit_charge)
        self._charge_h_converted = self._charge_h.convert_unit(self._unit_charge)
        unit_k = self._unit_energy * self._unit_length / self._unit_charge ** 2
        self._coulomb_k_converted = self._coulomb_k.convert_unit(unit_k)
        self.__c_o = self._charge_o_converted.value
        self.__c_h = self._charge_h_converted.value
        self.__k_e = self._coulomb_k_converted.value

        # Lennard-Jones
        self._lj_sigma_oo_converted = self._lj_sigma_oo.convert_unit(self._unit_length)
        self._lj_epsilon_oo_converted = self._lj_epsilon_oo.convert_unit(self._unit_energy)
        sigma_6 = self._lj_sigma_oo_converted ** 6
        self._lj_b = 4 * self._lj_epsilon_oo_converted * sigma_6
        self._lj_a = self._lj_b * sigma_6
        self.__lj_b = self._lj_b.value
        self.__lj_a = self._lj_a.value

        # Bond vibration
        unit_k = self._unit_energy / self._unit_length ** 2
        self._bond_k_converted = self._bond_k.convert_unit(unit_k)
        self._bond_eq_len_converted = self._bond_eq_len.convert_unit(self._unit_length)
        self.__k_b = self._bond_k_converted.value
        self.__eq_dist = self._bond_eq_len_converted.value

        # Angle vibration
        unit_angle = duq.Unit("rad")
        unit_k = self._unit_energy / unit_angle ** 2
        self._angle_k_converted = self._angle_k.convert_unit(unit_k)
        self._angle_eq_converted = self._angle_eq.convert_unit(unit_angle)
        self.__k_a = self._angle_k_converted.value
        self.__eq_angle = self._angle_eq_converted.value

        self._fitted = True
        return

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

            energy = (
                self.__k_e
                * self._charges[idx_curr_atom]
                * self._charges[idx_first_interacting_atom:]
                / dists
            )
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
            e_attractive = -self.__lj_b * inverse_dist_6
            e_repulsive = self.__lj_a * inverse_dist_6 ** 2
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
            dists = self._distances[idx_curr_atom, idx_curr_atom + 1 : idx_curr_atom + 3]
            dist_vectors = self._distance_vectors[
                idx_curr_atom, idx_curr_atom + 1 : idx_curr_atom + 3
            ]
            displacements = dists - self.__eq_dist
            k_times_displacements = self.__k_b * displacements

            self._energy_bond += (k_times_displacements * displacements / 2).sum()

            f = (-k_times_displacements / dists).reshape(-1, 1) * dist_vectors
            self._force[idx_curr_atom] += f.sum(axis=0)
            self._force[idx_curr_atom + 1 : idx_curr_atom + 3] += -f
        return

    def update_angle_vibration(self) -> None:

        for idx_curr_atom in range(0, self._num_atoms, 3):
            r_ml = -self._distance_vectors[idx_curr_atom, idx_curr_atom + 1]
            r_mr = -self._distance_vectors[idx_curr_atom, idx_curr_atom + 2]
            dist_ml = self._distances[idx_curr_atom, idx_curr_atom + 1]
            dist_mr = self._distances[idx_curr_atom, idx_curr_atom + 2]
            cos = np.dot(r_ml, r_mr) / (dist_ml * dist_mr)
            angle = np.arccos(cos)
            self._angles[idx_curr_atom // 3] = angle

            # Calculate common terms
            sin = np.sin(angle)
            angle_displacement = angle - self.__eq_angle
            a = self.__k_a * angle_displacement / sin
            dist_ml_mult_dist_mr = dist_ml * dist_mr
            r_ml_div_dist2_ml = r_ml / dist_ml ** 2
            r_mr_div_dist2_mr = r_mr / dist_mr ** 2

            # Calculate potential
            self._energy_angle += self.__k_a * angle_displacement ** 2

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
            Distance vectors and distances are stored in `self.distance_vectors` and `self.distances` respectively.

        Notes
        -----
        The distances 'q_i - q_j` are calculated for all unique pairs, where 'i' is smaller than 'j'.
        Distances 'q_i - q_j` where 'i' is larger than 'j' are equal to the negative value of 'q_j - q_i'.
        """

        for idx, coord in enumerate(self._q[:-1]):
            self._distance_vectors[idx, idx + 1 :] = coord - self._q[idx + 1 :]
        self._distances[...] = np.linalg.norm(self._distance_vectors, axis=2)
        return

    @property
    def model_parameters(self):
        str_repr = (
            f"{self._desc_charge_o} (q_O):\n"
            f"{self._charge_o.str_repr_short}"
            + (f" = {self._charge_o_converted.str_repr_short}\n" if self._fitted else "\n")
            + f"{self._desc_charge_h} (q_H):\n"
            f"{self._charge_h.str_repr_short}"
            + (f" = {self._charge_h_converted.str_repr_short}\n" if self._fitted else "\n")
            + f"{self._desc_lj_epsilon_oo} (ε_OO):\n"
            f"{self._lj_epsilon_oo.str_repr_short}"
            + (f" = {self._lj_epsilon_oo_converted.str_repr_short}\n" if self._fitted else "\n")
            + f"{self._desc_lj_sigma_oo} (σ_OO):\n"
            f"{self._lj_sigma_oo.str_repr_short}"
            + (
                f" = {self._lj_sigma_oo_converted.str_repr_short}\n"
                f"Lennard-Jones parameter A: {self._lj_a.str_repr_short}\n"
                f"Lennard-Jones parameter B: {self._lj_b.str_repr_short}\n"
                if self._fitted
                else "\n"
            )
            + f"{self._desc_bond_k} (k_bond):\n"
            f"{self._bond_k.str_repr_short}"
            + (f" = {self._bond_k_converted.str_repr_short}\n" if self._fitted else "\n")
            + f"{self._desc_bond_eq_len} (r_OH):\n"
            f"{self._bond_eq_len.str_repr_short}"
            + (f" = {self._bond_eq_len_converted.str_repr_short}\n" if self._fitted else "\n")
            + f"{self._desc_angle_k} (k_angle):\n"
            f"{self._angle_k.str_repr_short}"
            + (f" = {self._angle_k_converted.str_repr_short}\n" if self._fitted else "\n")
            + f"{self._desc_angle_eq} (θ_HOH):\n"
            f"{self._angle_eq.str_repr_short}"
            + (f" = {self._angle_eq_converted.str_repr_short}" if self._fitted else "")
        )
        return str_repr

    @property
    def dataframe(self):
        self._raise_for_model()
        return self._dataframe.loc[["Description", self._model_name]]

    @property
    def model_metadata(self):
        self._raise_for_model()
        str_repr = (
            f"Name: {self.model_name}\n"
            f"Description: {self.model_description}\n"
            f"Reference: {self.model_publication_name}, {self.model_publication_citation} {self.model_publication_link}"
        )
        return str_repr

    @property
    def model_name(self):
        self._raise_for_model()
        return self._model_name

    @property
    def model_description(self):
        self._raise_for_model()
        return self._model_description

    @property
    def model_publication_name(self):
        self._raise_for_model()
        return self._model_ref_name

    @property
    def model_publication_citation(self):
        self._raise_for_model()
        return self._model_ref_cite

    @property
    def model_publication_link(self):
        self._raise_for_model()
        return self._model_ref_link

    def model_publication_webpage(self):
        self._raise_for_model()
        webbrowser.open(self._model_ref_link)
        return

    def _raise_for_model(self):
        if self._model_name is None:
            raise ValueError("The force-field was not created from a model.")
