"""
This module contains the superclass `ForceField`, from which all force-field classes inherit.
"""

# Standard library
from __future__ import annotations
from typing import Tuple, Union
import webbrowser

# 3rd-party packages
import numpy as np
import pandas as pd
import duq

# Self
from .. import helpers


class ForceField:

    # Correct dimension of force-field parameters
    _dim_bond_vib_k = duq.Dimension("energy.length^-2")
    _dim_bond_eq_dist = duq.Dimension("length")
    _dim_angle_vib_k = duq.Dimension("energy.dimensionless^-2")
    _dim_angle_eq_angle = duq.Dimension("dimensionless")
    _dim_lj_epsilon = duq.Dimension("energy")
    _dim_lj_sigma = duq.Dimension("length")
    _dim_lj_a = duq.Dimension("energy.length^12")
    _dim_lj_b = duq.Dimension("energy.length^6")
    _dim_charge = duq.Dimension("electric charge")
    _dim_coulomb_k = duq.Dimension("energy.length.electric charge^-2")
    _dim_mass = duq.Dimension("mass")

    # Internal units of class data
    _unit_mass = duq.Unit("Da")
    _unit_charge = duq.Unit("e")

    # Pandas Dataframe containing several sets of parameters for the model.
    _dataframe = None

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
    def from_model(cls, model_name: str) -> ForceField:
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
            ForceField
            Instantiated object parametrized using the given parameters model.
        """
        pass

    def __init__(self):

        # Attributes for storing the data after each force-field update
        self._acceleration = None
        self._force_total = None
        self._force_coulomb = None
        self._force_lj = None
        self._force_bond = None
        self._force_angle = None
        self._distances = None
        self._distance_vectors = None
        self._angles = None
        self._energy_coulomb = 0
        self._energy_lj = 0
        self._energy_bond = 0
        self._energy_angle = 0

        # Attributes that are set after calling `initialize_forcefield`
        self._num_molecules = None
        self._num_atoms = None

        # Attributes that are only set when instantiating from alternative constructor `from_model`
        self._model_name = None
        self._model_description = None
        self._model_ref_name = None
        self._model_ref_cite = None
        self._model_ref_link = None
        return

    @property
    def acceleration(self) -> np.ndarray:
        return self._acceleration

    @property
    def force_total(self) -> np.ndarray:
        return self._force_total

    @property
    def force_coulomb(self):
        return self._force_coulomb

    @property
    def force_lennard_jones(self):
        return self._force_lj

    @property
    def force_bond_vibration(self):
        return self._force_bond

    @property
    def force_angle_vibration(self):
        return self._force_angle

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

    def __call__(self, positions: np.ndarray) -> None:
        self._update_distances(positions)
        self._update_forces_energies()
        self._update_acceleration()
        return

    def _update_forces_energies(self) -> None:
        self._update_coulomb()
        self._update_lennard_jones()
        self._update_bond_vibration()
        self._update_angle_vibration()
        self._force_total[...] = (
                self._force_coulomb + self._force_lj + self._force_bond + self._force_angle
        )
        return

    def _update_acceleration(self) -> None:
        pass

    def _update_coulomb(self) -> None:
        pass

    def _update_lennard_jones(self) -> None:
        pass

    def _update_bond_vibration(self) -> None:
        pass

    def _update_angle_vibration(self) -> None:
        pass

    def _update_distances(self, positions) -> None:
        pass

    def initialize_forcefield(self, shape_data) -> None:
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

        Returns
        -------
            None
            Arrays for storing the force-field evaluation results are initialized with the
            correct shape.
        """

        # Calculate number of atoms and molecules
        self._num_atoms = shape_data[0]
        self._num_molecules = self._num_atoms // 3

        # Initialize output arrays with the right shape
        self._acceleration = np.zeros(shape_data)
        self._force_total = np.zeros(shape_data)
        self._force_coulomb = np.zeros(shape_data)
        self._force_lj = np.zeros(shape_data)
        self._force_bond = np.zeros(shape_data)
        self._force_angle = np.zeros(shape_data)
        self._distances = np.zeros((self._num_atoms, self._num_atoms))
        self._distance_vectors = np.zeros((self._num_atoms, *shape_data))
        self._angles = np.zeros(self._num_molecules)
        return

    def _calculate_lj_params_a_b(
            self, lj_epsilon: Union[float, duq.Quantity], lj_sigma: Union[float, duq.Quantity]
    ) -> Tuple[Union[float, duq.Quantity], Union[float, duq.Quantity]]:
        """
        Calculate the Lennard-Jones parameters A and B, from epsilon and sigma.

        Parameters
        ----------
        lj_epsilon : Union[float, duq.Quantity]
            Lennard-Jones dispersion energy (ε), i.e. depth of the potential well.
        lj_sigma : Union[float, duq.Quantity]
            Lennard-Jones size of the particle (σ), i.e. the distance at which the potential is
            zero.

        Returns
        -------
        (lj_a, lj_b) : Tuple[Union[float, duq.Quantity], Union[float, duq.Quantity]]
            Lennard-Jones parameters A and B, either as floats or duq.Quantity objects, depending
            on the input arguments.
        """
        sigma_6 = lj_sigma ** 6
        lj_b = 4 * lj_epsilon * sigma_6
        lj_a = lj_b * sigma_6
        if isinstance(lj_a, duq.Quantity):
            helpers.raise_for_dimension(lj_a, self._dim_lj_a, "lj_a")
        if isinstance(lj_b, duq.Quantity):
            helpers.raise_for_dimension(lj_b, self._dim_lj_b, "lj_b")

        return lj_a, lj_b

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

