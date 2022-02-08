"""
This module contains the force-field of the flexible 3-site SPC water model.
"""


# Standard library
from __future__ import annotations
from typing import Union, Tuple
from pathlib import Path

# 3rd-party packages
import numpy as np
import pandas as pd
import duq

# Self
from ...data import atom_data
from ... import helpers
from ..forcefield_superclass import ForceField
from ... import terms_multi_vectorized_lazy as force_terms
from ... import switch_functions as switch


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
    """

    __slots__ = (
        "_k_b",
        "_k_b_conv",
        "__k_b",
        "_d0",
        "_d0_conv",
        "__d0",
        "_k_a",
        "_k_a_conv",
        "__k_a",
        "_angle0",
        "_angle0_conv",
        "__angle0",
        "_lj_epsilon",
        "_lj_epsilon_conv",
        "_lj_sigma",
        "_lj_sigma_conv",
        "_lj_a",
        "_lj_b",
        "_lj_a_conv",
        "_lj_b_conv",
        "__lj_a",
        "__lj_b",
        "_c_o",
        "_c_o_conv",
        "_c_h",
        "_c_h_conv",
        "__c",
        "_k_e",
        "_k_e_conv",
        "__k_e",
        "_m_o",
        "__m_o",
        "_m_h",
        "__m_h",
        "_unitless",
        "_fitted",
        "_unit_length",
        "_unit_time",
        "_unit_force",
        "_unit_energy",
    )

    # -- Setting class attributes --
    # Pandas Dataframe containing several sets of parameters for the model.
    _dataframe = pd.read_pickle(
        Path(__file__).parent.parent.parent / "data/model_params/water_flexible_3site_spc.pkl"
    )

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
            Instantiated object parametrized using the given parameters model.
        """
        # Read parameters (as strings) from the corresponding row in the dataframe
        p = "Parameters"
        charge_o = cls._dataframe.loc[model_name, (p, "Coulomb", "q_O")]
        charge_h = cls._dataframe.loc[model_name, (p, "Coulomb", "q_H")]
        lj_epsilon_oo = cls._dataframe.loc[model_name, (p, "Lennard-Jones", "ε_OO")]
        lj_sigma_oo = cls._dataframe.loc[model_name, (p, "Lennard-Jones", "σ_OO")]
        bond_k = cls._dataframe.loc[model_name, (p, "Bond vibration", "k")]
        bond_eq_len = cls._dataframe.loc[model_name, (p, "Bond vibration", "r_OH")]
        angle_k = cls._dataframe.loc[model_name, (p, "Angle vibration", "k")]
        angle_eq = cls._dataframe.loc[model_name, (p, "Angle vibration", "θ_HOH")]
        # Load constants
        mass_o = duq.Quantity(atom_data.mass_dalton[8], "Da")
        mass_h = duq.Quantity(atom_data.mass_dalton[1], "Da")
        coulomb_k = duq.predefined_constants.coulomb_const
        # Create `duq.Quantity` objects from the parameters and
        # instantiate the class using the main constructor.
        forcefield = cls(
            charge_oxygen=charge_o,
            charge_hydrogen=charge_h,
            coulomb_const=coulomb_k,
            lennard_jones_epsilon_oo=lj_epsilon_oo,
            lennard_jones_sigma_oo=lj_sigma_oo,
            bond_force_constant=bond_k,
            bond_eq_dist=bond_eq_len,
            angle_force_constant=angle_k,
            angle_eq_angle=angle_eq,
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
        """
        Instantiate the class by directly inputting all force-field parameters and constants.
        To instantiate the class using an available parameter-model, use the alternative
        constructor `Flexible3SiteSPC.from_model`.

        Parameters
        ----------
        bond_force_constant : Union[float, str, duq.Quantity]
            Force constant (k_b) of the harmonic O–H bond vibration potential, as a `duq.Quantity`
            object, or a string representing the value and unit, e.g. "1.059E+03 kcal.Å^-2.mol^-1".
        bond_eq_dist : Union[float, str, duq.Quantity]
            Equilibrium bond length of the O–H bond, as a `duq.Quantity` object, or a string
            representing the value and unit, e.g. "1.012 Å".
        angle_force_constant : Union[float, str, duq.Quantity]
            Force constant of the harmonic H–O–H angle vibration potential, as a `duq.Quantity`
            object, or a string representing the value and unit, e.g. "75.9 kcal.rad^-2.mol^-1".
            If the unit of angle is not provided, it is assumed to be in radian.
        angle_eq_angle : Union[float, str, duq.Quantity]
            Equilibrium H–O–H angle, as a `duq.Quantity` object, or a string representing the
            value and unit, e.g. "113.24 deg". If the unit of angle is not provided, it is assumed
            to be in radian.
        lennard_jones_epsilon_oo : Union[float, str, duq.Quantity]
            Lennard-Jones dispersion energy (ɛ_OO), i.e. the depth of the oxygen–oxygen potential
            well, as a `duq.Quantity` object, or a string representing the value and unit, e.g.
            "1.55E-01 kcal.mol^-1".
        lennard_jones_sigma_oo : Union[float, str, duq.Quantity]
            Lennard-Jones size of the particle (σ_OO), i.e. the oxygen–oxygen distance at which the
            potential is zero, as a `duq.Quantity` object, or a string representing the value and
            unit, e.g. "3.165 Å".
        charge_oxygen : Union[float, str, duq.Quantity]
            Electric charge of the oxygen atom, as a `duq.Quantity` object, or a string
            representing the value and unit, e.g. "-0.8 e".
        charge_hydrogen : Union[float, str, duq.Quantity]
            Electric charge of the hydrogen atom, as a `duq.Quantity` object, or a string
            representing the value and unit, e.g. "0.4 e".
        coulomb_const : Union[float, str, duq.Quantity]
        mass_oxygen : Union[float, str, duq.Quantity]
        mass_hydrogen : Union[float, str, duq.Quantity]
        """
        # Initialize superclass
        super().__init__()
        # Verify that arguments are either all numbers, or all strings/duq.Quantity
        args = list(locals().values())[1:-1]
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
        # Store parameters; if parameters are all unitless, store the numbers, otherwise, verify
        # that they have the correct dimensions, and store them as `duq.Quantity` objects.
        self._k_b = (
            bond_force_constant
            if self._unitless
            else helpers.convert_to_quantity(
                bond_force_constant, self._dim_k_b, "bond_force_constant"
            )
        )
        self._d0 = (
            bond_eq_dist
            if self._unitless
            else helpers.convert_to_quantity(bond_eq_dist, self._dim_d0, "bond_eq_dist")
        )
        self._k_a = (
            angle_force_constant
            if self._unitless
            else helpers.convert_to_quantity(
                angle_force_constant, self._dim_k_a, "angle_force_constant"
            )
        )
        self._angle0 = (
            angle_eq_angle
            if self._unitless
            else helpers.convert_to_quantity(angle_eq_angle, self._dim_angle0, "angle_eq_angle")
        )
        self._lj_epsilon = (
            lennard_jones_epsilon_oo
            if self._unitless
            else helpers.convert_to_quantity(
                lennard_jones_epsilon_oo, self._dim_lj_epsilon, "lennard_jones_epsilon_oo"
            )
        )
        self._lj_sigma = (
            lennard_jones_sigma_oo
            if self._unitless
            else helpers.convert_to_quantity(
                lennard_jones_sigma_oo, self._dim_lj_sigma, "lennard_jones_sigma_oo"
            )
        )
        self._c_o = (
            charge_oxygen
            if self._unitless
            else helpers.convert_to_quantity(charge_oxygen, self._dim_c, "charge_oxygen")
        )
        self._c_h = (
            charge_hydrogen
            if self._unitless
            else helpers.convert_to_quantity(charge_hydrogen, self._dim_c, "charge_hydrogen")
        )
        self._k_e = (
            coulomb_const
            if self._unitless
            else helpers.convert_to_quantity(coulomb_const, self._dim_k_e, "coulomb_const")
        )
        self._m_o = (
            mass_oxygen
            if self._unitless
            else helpers.convert_to_quantity(mass_oxygen, self._dim_m, "mass_oxygen")
        )
        self._m_h = (
            mass_hydrogen
            if self._unitless
            else helpers.convert_to_quantity(mass_hydrogen, self._dim_m, "mass_hydrogen")
        )
        # Calculate Lennard-Jones parameters A and B from epsilon and sigma
        self._lj_a, self._lj_b = helpers.calculate_lennard_jones_params_a_b(
            self._lj_epsilon, self._lj_sigma
        )
        # Attributes that are set after calling `fit_units_to_input_data`
        self._k_b_conv: duq.Quantity = None
        self._d0_conv: duq.Quantity = None
        self._k_a_conv: duq.Quantity = None
        self._angle0_conv: duq.Quantity = None
        self._lj_epsilon_conv: duq.Quantity = None
        self._lj_sigma_conv: duq.Quantity = None
        self._lj_a_conv: duq.Quantity = None
        self._lj_b_conv: duq.Quantity = None
        self._c_o_conv: duq.Quantity = None
        self._c_h_conv: duq.Quantity = None
        self._k_e_conv: duq.Quantity = None
        # Attributes holding numerical values of all parameters
        # If parameters were inputted as pure numbers, these are then directly set here,
        # otherwise they are set after calling the `fit_units_to_input_data` method.
        self.__k_b: float = self._k_b if self._unitless else None
        self.__d0: float = self._d0 if self._unitless else None
        self.__k_a: float = self._k_a if self._unitless else None
        self.__angle0: float = self._angle0 if self._unitless else None
        self.__lj_a: float = self._lj_a if self._unitless else None
        self.__lj_b: float = self._lj_b if self._unitless else None
        self.__k_e: float = self._k_e if self._unitless else None
        self.__m_o: float = self._m_o if self._unitless else None
        self.__m_h: float = self._m_h if self._unitless else None
        self.__c: np.ndarray = None  # This cannot be set now because number of atoms is needed
        return

    def initialize_forcefield(
        self, shape_data: Tuple[int, int], pbc_cell_lengths: np.ndarray = None
    ) -> None:
        """
        Prepare the force-field for a specific shape of input coordinates. This is necessary to
        determine the shape of arrays that are used to store the output data after each force
        evaluation, since these arrays are only created once and then overwritten after each
        re-evaluation.

        Parameters
        ----------
        shape_data : Tuple(int, int)
            Shape of the array of positions, where the first value is the number of atoms (should
            be a multiple of 3), and the second value is the number of spatial dimensions of the
            coordinates of each atom.
        pbc_cell_lengths : numpy.ndarray
            Lengths of the unit cell of the periodic system as a 1D-array of shape (3, ). If set to
            None, then periodic boundary condition will not be used.

        Returns
        -------
            None
            Arrays for storing the force-field evaluation results are initialized with the
            correct shape.
        """
        super().initialize_forcefield(shape_data, pbc_cell_lengths)
        if self._unitless:
            self.__c = np.tile([self._c_o, self._c_h, self._c_h], self._num_molecules)
        # Calculate index of first long-range interacting atom for each atom (i.e. the index of
        # first atom after the current atom that is not in the same molecule as the current atom)
        for idx_curr_atom in range(self._num_atoms - 3):
            self._idx_first_long_range_interacting_atom[idx_curr_atom] = (
                idx_curr_atom + 3 - idx_curr_atom % 3
            )
        return

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
        # Calculate and verify general units
        self._unit_force = self._unit_mass * self._unit_length / self._unit_time ** 2
        helpers.raise_for_dimension(self._unit_force, "force", "_unit_force")
        self._unit_energy = self._unit_force * self._unit_length
        helpers.raise_for_dimension(self._unit_energy, "energy", "_unit_energy")
        # Convert bond vibration units
        unit_bond_k = self._unit_energy / self._unit_length ** 2
        helpers.raise_for_dimension(unit_bond_k, self._dim_k_b, "unit_bond_k")
        self._k_b_conv = self._k_b.convert_unit(unit_bond_k)
        self._d0_conv = self._d0.convert_unit(self._unit_length)
        # Convert angle vibration units
        unit_angle_k = self._unit_energy / self._unit_angle ** 2
        helpers.raise_for_dimension(unit_angle_k, self._dim_k_a, "unit_angle_k")
        self._k_a_conv = self._k_a.convert_unit(unit_angle_k)
        self._angle0_conv = self._angle0.convert_unit(self._unit_angle)
        # Convert Lennard-Jones units
        self._lj_sigma_conv = self._lj_sigma.convert_unit(self._unit_length)
        self._lj_epsilon_conv = self._lj_epsilon.convert_unit(self._unit_energy)
        self._lj_a_conv = self._lj_a.convert_unit(self._unit_energy * self._unit_length ** 12)
        helpers.raise_for_dimension(self._lj_a_conv, self._dim_lj_a, "_lj_a_converted")
        self._lj_b_conv = self._lj_b.convert_unit(self._unit_energy * self._unit_length ** 6)
        helpers.raise_for_dimension(self._lj_b_conv, self._dim_lj_b, "_lj_b_converted")
        # Convert coulomb units
        unit_k_e = self._unit_energy * self._unit_length / self._unit_charge ** 2
        helpers.raise_for_dimension(unit_k_e, self._dim_k_e, "unit_k_e")
        self._k_e_conv = self._k_e.convert_unit(unit_k_e)
        self._c_o_conv = self._c_o.convert_unit(self._unit_charge)
        self._c_h_conv = self._c_h.convert_unit(self._unit_charge)
        # Assign numerical values of converted units to attributes used in force functions
        self.__k_b = self._k_b_conv.value
        self.__d0 = self._d0_conv.value
        self.__k_a = self._k_a_conv.value
        self.__angle0 = self._angle0_conv.value
        self.__lj_a = self._lj_a_conv.value
        self.__lj_b = self._lj_b_conv.value
        self.__k_e = self._k_e_conv.value
        # Create array of charge values in the correct unit
        # (for ease of vector calculation in Coulomb force evaluation)
        self.__c = np.tile(
            [
                self._c_o_conv.value,
                self._c_h_conv.value,
                self._c_h_conv.value,
            ],
            self._num_molecules,
        )
        self.__m_o = self._m_o.value
        self.__m_h = self._m_h.value
        self._fitted = True
        return

    def _update_acceleration(self):
        self._acceleration[::3] = self._force_total[::3] / self.__m_o
        self._acceleration[1::3] = self._force_total[1::3] / self.__m_h
        self._acceleration[2::3] = self._force_total[2::3] / self.__m_h
        return

    def _update_coulomb_pbc(self) -> None:
        pass

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
        for idx_curr_atom, idx_first_interacting_atom in enumerate(
            self._idx_first_long_range_interacting_atom
        ):
            # Retrieve the distance-vectors/distances between current atom and all
            # atoms after it, as two arrays
            q_jsi = self._distance_vectors[idx_curr_atom, idx_first_interacting_atom:]
            d_ijs = self._distances[idx_curr_atom, idx_first_interacting_atom:]
            # Calculate potentials and forces
            f_ijs, e_ijs = force_terms.coulomb(
                q_jsi,
                d_ijs,
                self.__c[idx_curr_atom] * self.__c[idx_first_interacting_atom:],
                self.__k_e
            )
            # Add the calculated values to the corresponding attributes
            self._energy_coulomb += e_ijs.sum()
            self._force_coulomb[idx_curr_atom] += f_ijs.sum(axis=0)
            self._force_coulomb[idx_first_interacting_atom:] += -f_ijs
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
            q_jsi = self._distance_vectors[idx_curr_atom, idx_first_interacting_atom::3]
            d_ijs = self._distances[idx_curr_atom, idx_first_interacting_atom::3]
            # Call the respective calculation function based on input specifications , i.e. either
            # `self._calculate_lennard_jones` or `self._calculate_lennard_jones_switch`
            f_ijs, e_ijs = self._func_calculate_lennard_jones(q_jsi, d_ijs)
            # Add the calculated values to the corresponding attributes
            self._energy_lj += e_ijs.sum()
            self._force_lj[idx_curr_atom] += f_ijs.sum(axis=0)
            self._force_lj[idx_first_interacting_atom::3] += -f_ijs
        return

    def _calculate_lennard_jones_switch(
        self, q_jsi: np.ndarray, d_ijs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the truncated Lennard-Jones potential and forces, using a switch function.

        Parameters
        ----------
        q_jsi : numpy.ndarray
        d_ijs : numpy.ndarray

        Returns
        -------

        """
        f_ijs, e_ijs = force_terms.lennard_jones_switch(
            q_jsi, d_ijs, self.__lj_a, self.__lj_b, self._switch
        )
        return f_ijs, e_ijs

    def _calculate_lennard_jones_full(
        self, q_jsi: np.ndarray, d_ijs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the complete non-truncated Lennard-Jones potential and forces.

        Parameters
        ----------
        q_jsi : numpy.ndarray
        d_ijs : numpy.ndarray

        Returns
        -------

        """
        f_ijs, e_ijs = force_terms.lennard_jones(q_jsi, d_ijs, self.__lj_a, self.__lj_b)
        return f_ijs, e_ijs

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
            q_jsi = self._distance_vectors[idx_curr_atom, idx_curr_atom + 1 : idx_curr_atom + 3]
            d_ijs = self._distances[idx_curr_atom, idx_curr_atom + 1 : idx_curr_atom + 3]
            # Calculate potentials and forces
            f_ijs, e_ijs = force_terms.bond_vibration_harmonic(q_jsi, d_ijs, self.__d0, self.__k_b)
            # Add the calculated values to the corresponding attributes
            self._energy_bond += e_ijs.sum()
            self._force_bond[idx_curr_atom] = f_ijs.sum(axis=0)
            self._force_bond[idx_curr_atom + 1 : idx_curr_atom + 3] = -f_ijs
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
            q_ji = -self._distance_vectors[idx_curr_atom, idx_curr_atom + 1]
            q_jk = -self._distance_vectors[idx_curr_atom, idx_curr_atom + 2]
            d_ij = self._distances[idx_curr_atom, idx_curr_atom + 1]
            d_jk = self._distances[idx_curr_atom, idx_curr_atom + 2]
            # Calculate potentials and forces
            f_i, f_j, f_k, e_ijk, angle = force_terms.angle_vibration_harmonic(
                q_ji, q_jk, d_ij, d_jk, self.__angle0, self.__k_a
            )
            # Add the calculated values to the corresponding attributes
            self._angles[idx_curr_atom // 3] = angle
            self._energy_angle += e_ijk
            self._force_angle[idx_curr_atom] = f_j
            self._force_angle[idx_curr_atom + 1] = f_i
            self._force_angle[idx_curr_atom + 2] = f_k
        return

    def __str__(self) -> str:
        """
        String representation of the force-field, containing information on the model (in case
        the force-field was instantiated using the alternative constructor method `from_model`),
        and all model-parameters in their given units, and (when the model has already been fitted
        to input data) the fitted units.
        """
        str_repr = (
            (
                f"Model Metadata:\n--------------\n{self.model_metadata}\n\n"
                if self._model_name is not None
                else ""
            )
            + f"Model Parameters:\n----------------\n"
            + (
                (
                    "(parameters have not yet been converted into the units of input data)\n"
                    if not self._fitted
                    else "(with converted values fitted to input data in parenthesis)\n"
                )
                if not self._unitless
                else "(parameters have been inputted without units)"
            )
            + self.model_parameters
        )
        return str_repr

    @property
    def model_parameters(self):
        # Read general parameter-descriptions from dataframe
        d, p = ("Description", "Parameters")
        descriptions = {
            "k_b": f"{self._dataframe.loc[d, (p, 'Bond vibration', 'k')]} (k_b)",
            "d0": f"{self._dataframe.loc[d, (p, 'Bond vibration', 'r_OH')]} (r_OH)",
            "k_a": f"{self._dataframe.loc[d, (p, 'Angle vibration', 'k')]} (k_a)",
            "angle0": f"{self._dataframe.loc[d, (p, 'Angle vibration', 'θ_HOH')]} (θ_HOH)",
            "lj_epsilon": f"{self._dataframe.loc[d, (p, 'Lennard-Jones', 'ε_OO')]} (ε_OO)",
            "lj_sigma": f"{self._dataframe.loc[d, (p, 'Lennard-Jones', 'σ_OO')]} (σ_OO)",
            "lj_a": "Lennard-Jones parameter A",
            "lj_b": "Lennard-Jones parameter B",
            "c_o": f"{self._dataframe.loc[d, (p, 'Coulomb', 'q_O')]} (q_O)",
            "c_h": f"{self._dataframe.loc[d, (p, 'Coulomb', 'q_H')]} (q_H)",
        }
        str_repr = ""
        for var_name, desc in descriptions.items():
            var = getattr(self, f"_{var_name}")
            var_conv = getattr(self, f"_{var_name}_conv") if self._fitted else None
            str_repr += f"{desc}: {var if self._unitless else var.str_repr_short}" + (
                f" (converted: {var_conv.str_repr_short})\n" if self._fitted else "\n"
            )
        return str_repr
