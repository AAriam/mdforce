"""
This module contains the superclass `ForceField`, from which all force-field classes inherit.
"""

# Standard library
from typing import Tuple, Union

# 3rd-party packages
import numpy as np
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
