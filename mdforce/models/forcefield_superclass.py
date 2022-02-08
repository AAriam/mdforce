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
from .. import distances, helpers
from ..data import param_data


class ForceField:

    __slots__ = (
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
        "_indices_first_long_range_interacting_atom",
        "_num_atoms",
        "_num_molecules",
        "_model_name",
        "_model_description",
        "_model_ref_name",
        "_model_ref_cite",
        "_model_ref_link",
        "_pbc_box_lengths",
        "_func_update_distances",
        "_func_calculate_lennard_jones",
        "_func_calculate_coulomb",
        "_fitted",
        "_unit_length",
        "_unit_time",
        "_unit_force",
        "_unit_energy",
        "_unitless",
    )

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
            print(model.model_metadata)
            print("Parameters:")
            print("----------")
            print(model.model_parameters)
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
        self._unitless: bool = None
        # Attributes for storing the data after each force-field update
        self._acceleration: np.ndarray = None
        self._force_total: np.ndarray = None
        self._force_coulomb: np.ndarray = None
        self._force_lj: np.ndarray = None
        self._force_bond: np.ndarray = None
        self._force_angle: np.ndarray = None
        self._distances: np.ndarray = None
        self._distance_vectors: np.ndarray = None
        self._angles: np.ndarray = None
        self._energy_coulomb: float = None
        self._energy_lj: float = None
        self._energy_bond: float = None
        self._energy_angle: float = None
        # Attributes that are set after calling `initialize_forcefield`
        self._num_molecules: int = None
        self._num_atoms: int = None
        self._pbc_box_lengths: np.ndarray = None
        self._func_update_distances: Callable = None
        self._func_calculate_lennard_jones: Callable = None
        self._func_calculate_coulomb: Callable = None
        # Attributes that are set after calling `fit_units_to_input_data`
        self._unit_length: duq.Unit = None
        self._unit_time: duq.Unit = None
        self._unit_force: duq.Unit = None
        self._unit_energy: duq.Unit = None
        self._fitted: bool = False  # Whether the model has been fitted to input data.

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
        # Calculate number of atoms and molecules
        self._num_atoms = shape_data[0]
        self._num_molecules = self._num_atoms // 3
        if pbc_cell_lengths is not None:
            self._pbc_box_lengths = pbc_cell_lengths
            self._func_update_distances = self._update_distances_pbc
            self._func_update_coulomb = self._update_coulomb_pbc_ewald
            self._func_update_lennard_jones = self._update_lennard_jones_pbc
        else:
            self._func_update_distances = self._update_distances
            self._func_update_coulomb = self._update_coulomb
            self._func_update_lennard_jones = self._update_lennard_jones
        self._initialize_output_arrays(shape_data)
        # Do other preparations specific to the force-field
        return

        self._model_name: str = None
        self._model_description: str = None
        self._model_ref_name: str = None
        self._model_ref_cite: str = None
        self._model_ref_link: str = None
        return

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
    def model_parameters(self):
        return

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

    @property
    def unit_energy(self) -> duq.Unit:
        return self._unit_energy

    @property
    def unit_length(self) -> duq.Unit:
        return self._unit_length

    @property
    def unit_angle(self) -> duq.Unit:
        return param_data.unit_angle

    @property
    def unit_time(self) -> duq.Unit:
        return self._unit_time

    def __call__(self, positions: np.ndarray) -> None:
        self._func_update_distances(positions)
        self._func_calculate_coulomb()
        self._update_lennard_jones()
        self._update_bond_vibration()
        self._update_angle_vibration()
        self._force_total[...] = (
            self._force_coulomb + self._force_lj + self._force_bond + self._force_angle
        )
        self._update_acceleration()
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
        pass

    def _initialize_output_arrays(self, shape_data) -> None:
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
        # Initialize output arrays with the right shape
        self._acceleration = np.zeros(shape_data)
        self._force_total = np.zeros(shape_data)
        self._force_coulomb = np.zeros(shape_data)
        self._force_lj = np.zeros(shape_data)
        self._force_bond = np.zeros(shape_data)
        self._force_angle = np.zeros(shape_data)
        self._angles = np.zeros(self._num_molecules)
        self._indices_first_long_range_interacting_atom = np.zeros(self._num_atoms - 3, dtype=int)
        return

    def _update_distances(self, positions: np.ndarray) -> None:
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
        self._distance_vectors, self._distances = distances.array_multi_self(positions)
        return

    def _update_distances_pbc(self, positions: np.ndarray) -> None:
        self._distance_vectors, self._distances = distances.array_multi_self_pbc(
            positions, self._indices_first_long_range_interacting_atom, self._pbc_box_lengths
        )
        return

    def _update_acceleration(self) -> None:
        pass

    def _update_coulomb(self) -> None:
        pass

    def _update_coulomb_pbc_ewald(self) -> None:
        pass

    def _update_coulomb_pbc_mesh_ewald(self) -> None:
        pass

    def _update_lennard_jones(self) -> None:
        pass

    def _calculate_lennard_jones_switch(
            self, q_jsi: np.ndarray, d_ijs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def _calculate_lennard_jones_full(
            self, q_jsi: np.ndarray, d_ijs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def _update_bond_vibration(self) -> None:
        pass

    def _update_angle_vibration(self) -> None:
        pass

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
