"""
This module contains the force-field of the flexible 3-site SPC water model.
"""

# Standard library
from __future__ import annotations
from typing import Union, Tuple
from pathlib import Path
import webbrowser

# 3rd-party packages
import numpy as np
import pandas as pd
import duq

# Self
from ...data import atom_data
from ... import helpers
from ..forcefield_superclass import ForceField


__all__ = ["Flexible3SiteSPC"]


class Flexible3SiteSPC(ForceField):
    """
    Class for constructing the force-field of the flexible 3-site SPC water model.
    This force-field is specifically implemented for input data that exclusively contains
    water molecules; the input positions-vector `q` must thus be a 2-dimensional numpy array
    of shape (n, m), where 'n' is the number of atoms (must be a multiple of 3), and 'm' is
    the number of spatial dimensions. Moreover, the coordinates must be ordered by the molecules,
    where the coordinates of the oxygen atom comes first. For example, assuming A_n_m denotes
    the array of coordinates of the mth A atom in nth molecule; then the input data must be
    in the form: [O_1_1, H_1_1, H_1_2, O_2_1, H_2_1, H_2_2, ...]

    Parameters
    ----------
    charge_oxygen : Union[str, duq.Quantity]
        Electric charge of the oxygen atom, as a `duq.Quantity` object,
        or a string representing the value and unit, e.g. "-0.8 e".
    charge_hydrogen : Union[str, duq.Quantity]
        Electric charge of the hydrogen atom, as a `duq.Quantity` object,
        or a string representing the value and unit, e.g. "0.4 e".
    lennard_jones_epsilon_oo : Union[str, duq.Quantity]
        Lennard-Jones dispersion energy (ɛ_OO), i.e. the depth of the oxygen–oxygen potential well,
        as a `duq.Quantity` object, or a string representing the value and unit,
        e.g. "1.55E-01 kcal.mol^-1".
    lennard_jones_sigma_oo : Union[str, duq.Quantity]
        Lennard-Jones size of the particle (σ_OO), i.e. the oxygen–oxygen distance at which the
        potential is zero, as a `duq.Quantity` object, or a string representing the value and unit,
        e.g. "3.165 Å".
    bond_force_constant : Union[str, duq.Quantity]
        Force constant (k_b) of the harmonic O–H bond vibration potential, as a `duq.Quantity`
        object, or a string representing the value and unit, e.g. "1.059E+03 kcal.Å^-2.mol^-1".
    bond_eq_dist : Union[str, duq.Quantity]
        Equilibrium bond length of the O–H bond, as a `duq.Quantity` object,
        or a string representing the value and unit, e.g. "1.012 Å".
    angle_force_constant : Union[str, duq.Quantity]
        Force constant of the harmonic H–O–H angle vibration potential, as a `duq.Quantity` object,
        or a string representing the value and unit, e.g. "75.9 kcal.rad^-2.mol^-1".
        If the unit of angle is not provided, it is assumed to be in radian.
    angle_eq_angle : Union[str, duq.Quantity]
        Equilibrium H–O–H angle, as a `duq.Quantity` object, or a string representing the
        value and unit, e.g. "113.24 deg".
        If the unit of angle is not provided, it is assumed to be in radian.
    """

    __slots__ = [
        "_distance_vectors",
        "_distances",
        "_acceleration",
        "_force_total",
        "_force_coulomb",
        "_force_lj",
        "_force_bond",
        "_force_angle",
        "_angles",
        "_energy_coulomb",
        "_energy_lj",
        "_energy_bond",
        "_energy_angle",
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
        "_lj_a_converted",
        "_lj_b_converted",
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
        "_unitless",
    ]

    # -- Setting class attributes --
    # Pandas Dataframe containing several sets of parameters for the model.
    _dataframe = pd.read_pickle(
        Path(__file__).parent.parent.parent / "data/model_params/water_flexible_3site_spc.pkl"
    )
    # Read general parameter-descriptions from dataframe
    d = "Description"
    p = "Parameters"
    _desc_charge_o = _dataframe.loc[d, (p, "Coulomb", "q_O")]
    _desc_charge_h = _dataframe.loc[d, (p, "Coulomb", "q_H")]
    _desc_lj_epsilon_oo = _dataframe.loc[d, (p, "Lennard-Jones", "ε_OO")]
    _desc_lj_sigma_oo = _dataframe.loc[d, (p, "Lennard-Jones", "σ_OO")]
    _desc_bond_k = _dataframe.loc[d, (p, "Bond vibration", "k")]
    _desc_bond_eq_len = _dataframe.loc[d, (p, "Bond vibration", "r_OH")]
    _desc_angle_k = _dataframe.loc[d, (p, "Angle vibration", "k")]
    _desc_angle_eq = _dataframe.loc[d, (p, "Angle vibration", "θ_HOH")]

    @classmethod
    def print_available_models(cls) -> None:
        """
        Print all the data on all available model parameters.

        Returns
        -------
            None
        """
        for name in cls._dataframe.index[1:]:
            model = cls.from_model(name)
            print(model.model_metadata + "\n\n" + str(model) + "\n")
        return

    @classmethod
    def from_model(cls, model_name: str) -> Flexible3SiteSPC:
        """
        Alternative constructor (factory) method for instantiating the class using an available
        set of model parameters.

        Parameters
        ----------
        model_name : str
            Name of the parameters model. Available models and their names can be viewed by calling
            the class method `print_available_models`, which prints all data.

        Returns
        -------
            Flexible3SiteSPC
            Instantiated `ForceField` object parametrized using the given parameters model.
        """

        # Read parameters (as strings) from the corresponding row in the dataframe,
        # and split in order to separate the value and unit.
        p = "Parameters"
        charge_o = cls._dataframe.loc[model_name, (p, "Coulomb", "q_O")].split()
        charge_h = cls._dataframe.loc[model_name, (p, "Coulomb", "q_H")].split()
        lj_epsilon_oo = cls._dataframe.loc[model_name, (p, "Lennard-Jones", "ε_OO")].split()
        lj_sigma_oo = cls._dataframe.loc[model_name, (p, "Lennard-Jones", "σ_OO")].split()
        bond_k = cls._dataframe.loc[model_name, (p, "Bond vibration", "k")].split()
        bond_eq_len = cls._dataframe.loc[model_name, (p, "Bond vibration", "r_OH")].split()
        angle_k = cls._dataframe.loc[model_name, (p, "Angle vibration", "k")].split()
        angle_eq = cls._dataframe.loc[model_name, (p, "Angle vibration", "θ_HOH")].split()

        # Load constants
        mass_o = duq.Quantity(atom_data.mass_dalton[8], "Da")
        mass_h = duq.Quantity(atom_data.mass_dalton[1], "Da")
        coulomb_k = duq.predefined_constants.coulomb_const

        # Create `duq.Quantity` objects from the parameters and
        # instantiate the class using the main constructor.
        forcefield = cls(
            charge_oxygen=duq.Quantity(float(charge_o[0]), charge_o[1]),
            charge_hydrogen=duq.Quantity(float(charge_h[0]), charge_h[1]),
            coulomb_const=coulomb_k,
            lennard_jones_epsilon_oo=duq.Quantity(float(lj_epsilon_oo[0]), lj_epsilon_oo[1]),
            lennard_jones_sigma_oo=duq.Quantity(float(lj_sigma_oo[0]), lj_sigma_oo[1]),
            bond_force_constant=duq.Quantity(float(bond_k[0]), bond_k[1]),
            bond_eq_dist=duq.Quantity(float(bond_eq_len[0]), bond_eq_len[1]),
            angle_force_constant=duq.Quantity(float(angle_k[0]), angle_k[1]),
            angle_eq_angle=duq.Quantity(float(angle_eq[0]), angle_eq[1]),
            mass_oxygen=mass_o,
            mass_hydrogen=mass_h,

        )

        # Set the parameters-model info
        forcefield._model_name = model_name
        m = "Metadata"
        forcefield._model_description = cls._dataframe.loc[model_name, (m, "Info", "Description")]
        forcefield._model_ref_name = cls._dataframe.loc[model_name, (m, "Reference", "Name")]
        forcefield._model_ref_cite = cls._dataframe.loc[model_name, (m, "Reference", "Citation")]
        forcefield._model_ref_link = cls._dataframe.loc[model_name, (m, "Reference", "Link")]
        # Return the instantiated object
        return forcefield

    def __init__(
        self,
        bond_force_constant: Union[float, str, duq.Quantity],
        bond_eq_dist: Union[float, str, duq.Quantity],
        angle_force_constant: Union[float, str, duq.Quantity],
        angle_eq_angle: Union[float, str, duq.Quantity],
        lennard_jones_epsilon_oo: Union[float, str, duq.Quantity],
        lennard_jones_sigma_oo: Union[float, str, duq.Quantity],
        charge_oxygen: Union[float, str, duq.Quantity],
        charge_hydrogen: Union[float, str, duq.Quantity],
        coulomb_const: Union[float, str, duq.Quantity] = None,
        mass_oxygen: Union[float, str, duq.Quantity] = None,
        mass_hydrogen: Union[float, str, duq.Quantity] = None,
    ):
        super().__init__()

        # Verify that arguments are either all numbers, or all strings/duq.Quantity
        args = list(locals().values())[1:]
        count_nums = 0
        for arg in args:
            if isinstance(arg, (int, float, np.number)):
                count_nums += 1
        if count_nums not in [0, len(args)]:
            raise ValueError("Either all or none of the arguments should be inputted as numbers.")
        elif count_nums == 0:
            self._unitless = False
        else:
            self._unitless = True

        # Store parameters
        self._bond_k = (
            bond_force_constant if self._unitless else helpers.convert_to_quantity(
                            bond_force_constant, self._dim_bond_vib_k, "bond_force_constant"
            )
        )
        self._bond_eq_len = (
            bond_eq_dist if self._unitless else helpers.convert_to_quantity(
                bond_eq_dist, self._dim_bond_eq_dist, "bond_eq_dist"
            )
        )
        self._angle_k = (
            angle_force_constant if self._unitless else helpers.convert_to_quantity(
                angle_force_constant, self._dim_angle_vib_k, "angle_force_constant"
            )
        )
        self._angle_eq = (
            angle_eq_angle if self._unitless else helpers.convert_to_quantity(
                angle_eq_angle, self._dim_angle_eq_angle, "angle_eq_angle"
            )
        )
        self._lj_epsilon_oo = (
            lennard_jones_epsilon_oo if self._unitless else helpers.convert_to_quantity(
                lennard_jones_epsilon_oo, self._dim_lj_epsilon, "lennard_jones_epsilon_oo"
            )
        )
        self._lj_sigma_oo = (
            lennard_jones_sigma_oo if self._unitless else helpers.convert_to_quantity(
                lennard_jones_sigma_oo, self._dim_lj_sigma, "lennard_jones_sigma_oo"
            )
        )
        self._charge_o = (
            charge_oxygen if self._unitless else helpers.convert_to_quantity(
                charge_oxygen, self._dim_charge, "charge_oxygen"
            )
        )
        self._charge_h = (
            charge_hydrogen if self._unitless else helpers.convert_to_quantity(
                charge_hydrogen, self._dim_charge, "charge_hydrogen"
            )
        )
        self._coulomb_k = (
            coulomb_const if self._unitless else helpers.convert_to_quantity(
                coulomb_const, self._dim_coulomb_k, "coulomb_const"
            )
        )
        self._mass_o = (
            mass_oxygen if self._unitless else helpers.convert_to_quantity(
                mass_oxygen, self._dim_mass, "mass_oxygen"
            )
        )
        self._mass_h = (
            mass_hydrogen if self._unitless else helpers.convert_to_quantity(
                mass_hydrogen, self._dim_mass, "mass_hydrogen"
            )
        )

        # Calculate Lennard-Jones parameters A and B from epsilon and sigma
        self._lj_a, self._lj_b = self._calculate_lj_params_a_b(
            self._lj_epsilon_oo, self._lj_sigma_oo
        )

        # Attributes that are only set when instantiating from alternative constructor `from_model`
        self._model_name = None
        self._model_description = None
        self._model_ref_name = None
        self._model_ref_cite = None
        self._model_ref_link = None

        # Attributes that are set after calling `fit_units_to_input_data`
        self._charges = None
        self._coulomb_k_converted = None
        self._charge_o_converted = None
        self._charge_h_converted = None
        self._lj_epsilon_oo_converted = None
        self._lj_sigma_oo_converted = None
        self._lj_a_converted = None
        self._lj_b_converted = None
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

    def fit_units_to_input_data(
            self,
            unit_length: Union[str, duq.Unit],
            unit_time: Union[str, duq.Unit],
    ) -> None:
        """
        Make the force-field parameters' units compatible to those of the input data.
        This is only applicable when the force-field parameters are not inputted as pure numbers.

        Parameters
        ----------
        unit_length : Union[str, duq.Unit]
            Unit of length used in the data (i.e. positions and velocities), either as a `duq.Unit`
            object or a string representation of the unit, e.g. "Å".
        unit_time : Union[str, duq.Unit]
            Unit of time used in the data (i.e. velocities) and in the integrator, either as a
            `duq.Unit` object or a string representation of the unit, e.g. "Å".

        Returns
        -------
            None
            All force-field parameters are converted to be compatible with the given units.
        """

        if self._unitless:
            raise ValueError("Input parameters were inputted as unitless numbers.")
        else:
            self._unit_length = helpers.convert_to_unit(unit_length, "length", "unit_length")
            self._unit_time = helpers.convert_to_unit(unit_time, "time", "unit_time")

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
        self._lj_a_converted = self._lj_a.convert_unit()
        self._lj_b = self._lj_b.convert_unit()
        self.__lj_a = self._lj_a_converted.value
        self.__lj_b = self._lj_b_converted.value

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

        # Create array of charge values in the correct unit
        # (for ease of vector calculation in Coulomb force evaluation)
        self._charges = np.tile(
            [
                self._charge_o_converted.value,
                self._charge_h_converted.value,
                self._charge_h_converted.value,
            ],
            self._num_molecules,
        )

        self._fitted = True
        return

    def _update_acceleration(self):
        self._acceleration[::3] = self._force_total[::3] / self._mass_o
        self._acceleration[1::3] = self._force_total[1::3] / self._mass_h
        self._acceleration[2::3] = self._force_total[2::3] / self._mass_h
        return

    def _update_coulomb(self) -> None:
        """
        Calculate the Coulomb potential of the system, and the force vector on each atom.

        Returns
        -------
            None

        Notes
        -----
        The results of calculations are stored in the corresponding instance attributes:
            `self._force_coulomb`: stores the force vectors for each atom,
            due to the Coulomb potential.
            `self._energy_coulomb`: stores the total Coulomb potential of the system.
        """

        # Reset attributes that store the calculated values
        self._force_coulomb[...] = 0
        self._energy_coulomb = 0

        # Iterate over the indices of all atoms, other than the last three ones
        for idx_curr_atom in range(self._num_atoms - 3):
            idx_first_interacting_atom = idx_curr_atom + 3 - idx_curr_atom % 3

            # Retrieve the distance-vectors/distances between current atom and all
            # atoms after it, as two arrays
            dists = self._distances[idx_curr_atom, idx_first_interacting_atom:]
            dist_vectors = self._distance_vectors[idx_curr_atom, idx_first_interacting_atom:]

            # Calculate the potential between current atom and all atoms after it
            energy = (
                self.__k_e
                * self._charges[idx_curr_atom]
                * self._charges[idx_first_interacting_atom:]
                / dists
            )
            self._energy_coulomb += energy.sum()

            # Calculate the force on all atoms, using the calculated potential
            f = (energy / dists ** 2).reshape(-1, 1) * dist_vectors
            self._force_coulomb[idx_curr_atom] += f.sum(axis=0)
            self._force_coulomb[idx_first_interacting_atom:] += -f
        return

    def _update_lennard_jones(self) -> None:
        """
        Calculate the Lennard-Jones potential of the system, and the force vector on each atom.

        Returns
        -------
            None

        Notes
        -----
        The results of calculations are stored in the corresponding instance attributes:
            `self._force_lj`: stores the force vectors for each atom, due to the LJ-potential.
            `self._energy_lj`: stores the total LJ-potential of the system.
        """

        # Reset attributes that store the calculated values
        self._force_lj[...] = 0
        self._energy_lj = 0

        # Iterate over the indices of all oxygen atoms, other than the last one
        for idx_curr_atom in range(0, self._num_atoms - 3, 3):

            # Calculate index of first interacting atom, i.e. the next oxygen
            idx_first_interacting_atom = idx_curr_atom + 3
            # Retrieve the distance-vectors/distances between current oxygen and all oxygen
            # atoms after it, as two arrays
            dists = self._distances[idx_curr_atom, idx_first_interacting_atom::3]
            dist_vectors = self._distance_vectors[idx_curr_atom, idx_first_interacting_atom::3]

            # Calculate common terms only once
            inverse_dist_2 = 1 / dists ** 2
            inverse_dist_6 = inverse_dist_2 ** 3

            # Calculate the potential between current oxygen and all oxygen atoms after it
            e_attractive = -self.__lj_b * inverse_dist_6
            e_repulsive = self.__lj_a * inverse_dist_6 ** 2
            self._energy_lj += (e_repulsive + e_attractive).sum()

            # Calculate the force on all oxygen atoms, using the calculated potential
            f_attractive = 6 * e_attractive
            f_repulsive = 12 * e_repulsive
            f = ((f_attractive + f_repulsive) * inverse_dist_2).reshape(-1, 1) * dist_vectors
            self._force_lj[idx_curr_atom] += f.sum(axis=0)
            self._force_lj[idx_first_interacting_atom::3] += -f
        return

    def _update_bond_vibration(self) -> None:
        """
        Calculate the harmonic bond-vibration potential of the system, and the force vector on
        each atom.

        Returns
        -------
            None

        Notes
        -----
        The results of calculations are stored in the corresponding instance attributes:
            `self._force_bond`: stores the force vectors for each atom, due to bond vibration
            `self._energy_bond`: stores the total potential of the system due to bond vibration.
        """

        # Reset attributes that store the calculated values
        # In this case, there is no need to reset the array `self._force_bond`
        # since each array element is only written once, so it can be simply overwritten.
        self._energy_bond = 0

        # Iterate over the indices of all oxygen atoms
        for idx_curr_atom in range(0, self._num_atoms, 3):

            # Retrieve the distance-vectors/distances between the oxygen and the two hydrogen
            # atoms in the same molecule (these are the next two atoms after each oxygen)
            # as two single arrays
            dist_vectors = self._distance_vectors[
                           idx_curr_atom, idx_curr_atom + 1: idx_curr_atom + 3
                           ]
            dists = self._distances[idx_curr_atom, idx_curr_atom + 1 : idx_curr_atom + 3]

            # Calculate common terms only once
            displacements = dists - self.__eq_dist
            k_times_displacements = self.__k_b * displacements

            # Calculate the potential of the whole molecule
            self._energy_bond += (k_times_displacements * displacements / 2).sum()

            # Calculate forces on each atom
            f = (-k_times_displacements / dists).reshape(-1, 1) * dist_vectors
            self._force_coulomb[idx_curr_atom] = f.sum(axis=0)
            self._force_coulomb[idx_curr_atom + 1 : idx_curr_atom + 3] = -f
        return

    def _update_angle_vibration(self) -> None:
        """
        Calculate the harmonic angle-vibration potential of the system, and the force vector on
        each atom.

        Returns
        -------
            None

        Notes
        -----
        The results of calculations are stored in the corresponding instance attributes:
            `self._force_angle`: stores the force vectors for each atom, due to angle vibration
            `self._energy_angle`: stores the total potential of the system due to angle vibration.
            `self._angles`: stores the calculated angle for each molecule.
        """

        # Reset attributes that store the calculated values
        # In this case, there is no need to reset `self._angles` and `self._force_angle` arrays
        # since each array element is only written once, so it can be simply overwritten.
        self._energy_angle = 0

        # Iterate over the indices of all oxygen atoms
        for idx_curr_atom in range(0, self._num_atoms, 3):

            # Retrieve the distance-vectors/distances between the oxygen, and the two hydrogen
            # atoms in the same molecule (these are the next two atoms after each oxygen)
            # as two separate arrays/two separate values.
            r_ml = -self._distance_vectors[idx_curr_atom, idx_curr_atom + 1]
            r_mr = -self._distance_vectors[idx_curr_atom, idx_curr_atom + 2]
            dist_ml = self._distances[idx_curr_atom, idx_curr_atom + 1]
            dist_mr = self._distances[idx_curr_atom, idx_curr_atom + 2]

            # Calculate the angle from the dot product formula
            cos = np.dot(r_ml, r_mr) / (dist_ml * dist_mr)
            angle = np.arccos(cos)
            # Store the angle
            self._angles[idx_curr_atom // 3] = angle

            # Calculate common terms only once
            sin = np.sin(angle)
            angle_displacement = angle - self.__eq_angle
            a = self.__k_a * angle_displacement / sin
            dist_ml_mult_dist_mr = dist_ml * dist_mr
            r_ml_div_dist2_ml = r_ml / dist_ml ** 2
            r_mr_div_dist2_mr = r_mr / dist_mr ** 2

            # Calculate the potential of the whole molecule
            self._energy_angle += self.__k_a * angle_displacement ** 2

            # Calculate the force on oxygen
            b1 = -(r_ml + r_mr) / dist_ml_mult_dist_mr
            c1 = cos * (r_ml_div_dist2_ml + r_mr_div_dist2_mr)
            self._force_angle[idx_curr_atom] = a * (b1 + c1)

            # Calculate the force on first hydrogen
            b2 = r_mr / dist_ml_mult_dist_mr
            c2 = cos * r_ml_div_dist2_ml
            self._force_angle[idx_curr_atom + 1] = a * (b2 - c2)

            # Calculate the force on second hydrogen
            b3 = r_ml / dist_ml_mult_dist_mr
            c3 = cos * r_mr_div_dist2_mr
            self._force_angle[idx_curr_atom + 2] = a * (b3 - c3)
        return

    def _update_distances(self, positions) -> None:
        """
        Calculate the distance vector and distance between all unique pairs of atoms
        at the current state.

        Returns
        -------
            None
            Distance vectors and distances are stored in `self.distance_vectors` and
            `self.distances` respectively.

        Notes
        -----
        The distance vectors 'q_i - q_j' and their corresponding norms '||q_i - q_j||' are
        calculated for all unique pairs, where 'i' is smaller than 'j', and are accessed by
        `self._distance_vectors[i, j]` and `self._distances[i, j]`. However, for the case where
        'i' is larger than 'j', instead of calling `self._distance_vectors[i, j]` and
        `self._distances[i, j]`, `-self._distance_vectors[j, i]` and
        `-self._distances[j, i]` should be called, respectively (notice the negative sign in the
        beginning).
        """

        # Iterate over all atoms (other than the last atom)
        for idx_atom, coord_atom in enumerate(positions[:-1]):
            # Calculate distance vectors between that atom and all other atoms after it
            self._distance_vectors[idx_atom, idx_atom + 1 :] = coord_atom - positions[idx_atom + 1 :]
        # Calculate all distances at once, from the distance vectors
        self._distances[...] = np.linalg.norm(self._distance_vectors, axis=2)
        return

    def __str__(self) -> str:
        """
        String representation of the force-field, containing information on the model (in case
        the force-field was instantiated using the alternative constructor method `from_model`),
        and all model-parameters in their given units, and (when the model has already been fitted
        to input data) the fitted units.
        """

        str_repr = (
            (f"Model Metadata:\n--------------\n{self.model_metadata}\n\n"
             if self._model_name is not None else "") +
            f"Model Parameters:\n----------------\n" +
            ("(parameters have not yet been converted into the units of input data)\n"
             if not self._fitted else
             "(with converted values fitted to input data after equal sign)\n") +
            f"\t{self._desc_charge_o} (q_O):\n"
            f"{self._charge_o.str_repr_short}"
            + (f" = {self._charge_o_converted.str_repr_short}\n" if self._fitted else "\n")
            + f"\t{self._desc_charge_h} (q_H):\n"
            f"{self._charge_h.str_repr_short}"
            + (f" = {self._charge_h_converted.str_repr_short}\n" if self._fitted else "\n")
            + f"\t{self._desc_lj_epsilon_oo} (ε_OO):\n"
            f"{self._lj_epsilon_oo.str_repr_short}"
            + (f" = {self._lj_epsilon_oo_converted.str_repr_short}\n" if self._fitted else "\n")
            + f"\t{self._desc_lj_sigma_oo} (σ_OO):\n"
            f"{self._lj_sigma_oo.str_repr_short}"
            + (
                f" = {self._lj_sigma_oo_converted.str_repr_short}\n"
                f"\tLennard-Jones parameter A: {self._lj_a.str_repr_short}\n"
                f"\tLennard-Jones parameter B: {self._lj_b.str_repr_short}\n"
                if self._fitted
                else "\n"
            )
            + f"\t{self._desc_bond_k} (k_bond):\n"
            f"{self._bond_k.str_repr_short}"
            + (f" = {self._bond_k_converted.str_repr_short}\n" if self._fitted else "\n")
            + f"\t{self._desc_bond_eq_len} (r_OH):\n"
            f"{self._bond_eq_len.str_repr_short}"
            + (f" = {self._bond_eq_len_converted.str_repr_short}\n" if self._fitted else "\n")
            + f"\t{self._desc_angle_k} (k_angle):\n"
            f"{self._angle_k.str_repr_short}"
            + (f" = {self._angle_k_converted.str_repr_short}\n" if self._fitted else "\n")
            + f"\t{self._desc_angle_eq} (θ_HOH):\n"
            f"{self._angle_eq.str_repr_short}"
            + (f" = {self._angle_eq_converted.str_repr_short}" if self._fitted else "")
        )
        return str_repr

    @property
    def model_dataframe(self) -> pd.DataFrame:
        """
        All data on the parameters-model used in the force-field, as a Pandas.DataFrame object.

        Returns
        -------
            pd.DataFrame

        Raises
        ------
        ValueError
            When the `ForceField` object is not instantiated using the
            alternative constructor method `from_model` (and thus has no model data).
        """

        self._raise_for_model()
        return self._dataframe.loc[["Description", self._model_name]]

    @property
    def model_metadata(self) -> str:
        """
        Metadata of the parameters-model used in the force-field, including its name, description,
        and citation reference.

        Returns
        -------
            str

        Raises
        ------
        ValueError
            When the `ForceField` object is not instantiated using the
            alternative constructor method `from_model` (and thus has no model data).
        """

        self._raise_for_model()
        str_repr = (
            f"Name: {self.model_name}\n"
            f"Description: {self.model_description}\n"
            f"Reference: {self.model_publication_name}, "
            f"{self.model_publication_citation} {self.model_publication_link}"
        )
        return str_repr

    @property
    def model_name(self) -> str:
        """
        Name of the parameters-model used in the force-field.

        Returns
        -------
            str

        Raises
        ------
        ValueError
            When the `ForceField` object is not instantiated using the
            alternative constructor method `from_model` (and thus has no model data).
        """

        self._raise_for_model()
        return self._model_name

    @property
    def model_description(self) -> str:
        """
        Description of the parameters-model used in the force-field.

        Returns
        -------
            str

        Raises
        ------
        ValueError
            When the `ForceField` object is not instantiated using the
            alternative constructor method `from_model` (and thus has no model data).
        """

        self._raise_for_model()
        return self._model_description

    @property
    def model_publication_name(self) -> str:
        """
        Name of the publication for the parameters-model used in the force-field.

        Returns
        -------
            str

        Raises
        ------
        ValueError
            When the `ForceField` object is not instantiated using the
            alternative constructor method `from_model` (and thus has no model data).
        """

        self._raise_for_model()
        return self._model_ref_name

    @property
    def model_publication_citation(self) -> str:
        """
        Citation reference of publication for the parameters-model used in the force-field.

        Returns
        -------
            str

        Raises
        ------
        ValueError
            When the `ForceField` object is not instantiated using the
            alternative constructor method `from_model` (and thus has no model data).
        """

        self._raise_for_model()
        return self._model_ref_cite

    @property
    def model_publication_link(self) -> str:
        """
        Hyperlink of the publication for the parameters-model used in the force-field.

        Returns
        -------
            str

        Raises
        ------
        ValueError
            When the `ForceField` object is not instantiated using the
            alternative constructor method `from_model` (and thus has no model data).
        """

        self._raise_for_model()
        return self._model_ref_link

    def model_publication_webpage(self) -> None:
        """
        Open the webpage of the publication for the parameters-model in the default browser.

        Returns
        -------
            bool

        Raises
        ------
        ValueError
            When the `ForceField` object is not instantiated using the
            alternative constructor method `from_model` (and thus has no model data).
        """

        self._raise_for_model()
        webbrowser.open_new(self._model_ref_link)
        return

    def _raise_for_model(self) -> None:
        """
        Method used by all properties/methods corresponding to model-data, to raise an error when
        the model-data is not available (because the object was not instantiated using the
        alternative constructor method `from_model`).

        Returns
        -------
            None

        Raises
        ------
        ValueError
        """

        if self._model_name is None:
            raise ValueError("The force-field was not created from a parameter model.")
        return
