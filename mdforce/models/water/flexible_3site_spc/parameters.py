"""
Module containing the ModelParameters class for storing model parameters.
"""

# Standard library
from pathlib import Path
import webbrowser

# 3rd-party packages
import pandas as pd
from duq.unit import Unit
from duq.quantity import Quantity
from duq.quantity import predefined as consts

# Self
from ....element_masses import masses


class ModelParameters:
    """
    Class for storing parameters of a certain flexible-3site-SPC water model.

    Parameters
    ----------
    model_name : str
        Name of the model from the dataframe.
    """

    _dataframe = pd.read_pickle(Path(__file__).parent / "params.pkl")

    @classmethod
    def available_models(cls):
        for name in cls._dataframe.index[1:]:
            model = cls(name)
            print(model.model_metadata + "\n")

    def __init__(self, model_name):

        self._name = model_name

        self._mass_o = Quantity(masses[8], "Da")
        self._mass_o_converted = None
        self._mass_h = Quantity(masses[1], "Da")
        self._mass_h_converted = None

        self._coulomb_k = consts.coulomb_const
        self._coulomb_k_converted = None

        self._desc_charge_o = self._dataframe.loc["Description", ("Parameters", "Coulomb", "q_O")]
        self._desc_charge_h = self._dataframe.loc["Description", ("Parameters", "Coulomb", "q_H")]
        self._desc_lj_epsilon_oo = self._dataframe.loc["Description", ("Parameters", "Lennard-Jones", "ε_OO")]
        self._desc_lj_sigma_oo = self._dataframe.loc["Description", ("Parameters", "Lennard-Jones", "σ_OO")]
        self._desc_bond_k_oh = self._dataframe.loc["Description", ("Parameters", "Bond vibration", "k")]
        self._desc_bond_eq_len_oh = self._dataframe.loc["Description", ("Parameters", "Bond vibration", "r_OH")]
        self._desc_angle_k_hoh = self._dataframe.loc["Description", ("Parameters", "Angle vibration", "k")]
        self._desc_angle_eq_hoh = self._dataframe.loc["Description", ("Parameters", "Angle vibration", "θ_HOH")]
        self._description = self._dataframe.loc[model_name, ("Metadata", "Info", "Description")]
        self._ref_name = self._dataframe.loc[model_name, ("Metadata", "Reference", "Name")]
        self._ref_cite = self._dataframe.loc[model_name, ("Metadata", "Reference", "Citation")]
        self._ref_link = self._dataframe.loc[model_name, ("Metadata", "Reference", "Link")]

        charge_o = self._dataframe.loc[model_name, ("Parameters", "Coulomb", "q_O")].split()
        charge_h = self._dataframe.loc[model_name, ("Parameters", "Coulomb", "q_H")].split()
        lj_epsilon_oo = self._dataframe.loc[model_name, ("Parameters", "Lennard-Jones", "ε_OO")].split()
        lj_sigma_oo = self._dataframe.loc[model_name, ("Parameters", "Lennard-Jones", "σ_OO")].split()
        bond_k_oh = self._dataframe.loc[model_name, ("Parameters", "Bond vibration", "k")].split()
        bond_eq_len_oh = self._dataframe.loc[model_name, ("Parameters", "Bond vibration", "r_OH")].split()
        angle_k_hoh = self._dataframe.loc[model_name, ("Parameters", "Angle vibration", "k")].split()
        angle_eq_hoh = self._dataframe.loc[model_name, ("Parameters", "Angle vibration", "θ_HOH")].split()

        self._charge_o = Quantity(float(charge_o[0]), charge_o[1])
        self._charge_o_converted = None
        self._charge_h = Quantity(float(charge_h[0]), charge_h[1])
        self._charge_h_converted = None
        self._lj_epsilon_oo = Quantity(float(lj_epsilon_oo[0]), lj_epsilon_oo[1])
        self._lj_epsilon_oo_converted = None
        self._lj_sigma_oo = Quantity(float(lj_sigma_oo[0]), lj_sigma_oo[1])
        self._lj_sigma_oo_converted = None
        self._lj_b = None
        self._lj_a = None
        self._bond_k_oh = Quantity(float(bond_k_oh[0]), bond_k_oh[1])
        self._bond_k_oh_converted = None
        self._bond_eq_len_oh = Quantity(float(bond_eq_len_oh[0]), bond_eq_len_oh[1])
        self._angle_eq_hoh_converted = None
        self._angle_k_hoh = Quantity(float(angle_k_hoh[0]), angle_k_hoh[1])
        self._angle_k_hoh_converted = None
        self._angle_eq_hoh = Quantity(float(angle_eq_hoh[0]), angle_eq_hoh[1])
        self._bond_eq_len_oh_converted = None

        self._unit_charge = None
        self._unit_mass = None
        self._unit_length = None
        self._unit_time = None
        self._unit_force = None
        self._unit_energy = None

    def unify_units(self, unit_length, unit_time, unit_mass="Da", unit_charge="e"):
        self._unit_mass = Unit(unit_mass)
        self._unit_length = Unit(unit_length)
        self._unit_time = Unit(unit_time)
        self._unit_charge = Unit(unit_charge)
        
        self._unit_force = self._unit_mass * self._unit_length / self._unit_time ** 2
        self._unit_energy = self._unit_force * self._unit_length
        
        self.unify_units_mass()
        self.unify_units_coulomb()
        self.unify_units_lennard_jones()
        self.unify_units_bond_vibration()
        self.unify_units_angle_vibration()
        return

    def unify_units_mass(self):
        self._mass_o_converted = self._mass_o.convert_unit(self._unit_mass)
        self._mass_h_converted = self._mass_h.convert_unit(self._unit_mass)
        return

    def unify_units_coulomb(self):
        self._charge_o_converted = self._charge_o.convert_unit(self._unit_charge)
        self._charge_h_converted = self._charge_h.convert_unit(self._unit_charge)
        unit_k = self._unit_energy * self._unit_length / self._unit_charge ** 2
        self._coulomb_k_converted = self._coulomb_k.convert_unit(unit_k)
        return

    def unify_units_lennard_jones(self):
        self._lj_sigma_oo_converted = self._lj_sigma_oo.convert_unit(self._unit_length)
        self._lj_epsilon_oo_converted = self._lj_epsilon_oo.convert_unit(self._unit_energy)
        
        sigma_6 = self._lj_sigma_oo_converted ** 6
        self._lj_b = 4 * self._lj_epsilon_oo_converted * sigma_6
        self._lj_a = self._lj_b * sigma_6
        return
    
    def unify_units_bond_vibration(self):
        unit_k = self._unit_energy / self._unit_length ** 2
        self._bond_k_oh_converted = self._bond_k_oh.convert_unit(unit_k)
        self._bond_eq_len_oh_converted = self._bond_eq_len_oh.convert_unit(self._unit_length)
        return
    
    def unify_units_angle_vibration(self):
        unit_angle = Unit("rad")
        unit_k = self._unit_energy / unit_angle ** 2
        self._angle_k_hoh_converted = self._angle_k_hoh.convert_unit(unit_k)
        self._angle_eq_hoh_converted = self._angle_eq_hoh.convert_unit(unit_angle)
        return
    
    @property
    def dataframe(self):
        return self._dataframe.loc[["Description", self._name]]

    @property
    def model_parameters(self):
        str_repr = (
            f"{self._desc_charge_o} (q_O):\n"
            f"{self._charge_o.str_repr_short} = {self._charge_o_converted.str_repr_short}\n"
            f"{self._desc_charge_h} (q_H):\n"
            f"{self._charge_h.str_repr_short} = {self._charge_h_converted.str_repr_short}\n"
            f"{self._desc_lj_epsilon_oo} (ε_OO):\n"
            f"{self._lj_epsilon_oo.str_repr_short} = {self._lj_epsilon_oo_converted.str_repr_short}\n"
            f"{self._desc_lj_sigma_oo} (σ_OO):\n"
            f"{self._lj_sigma_oo.str_repr_short} = {self._lj_sigma_oo_converted.str_repr_short}\n"
            f"Lennard-Jones parameter A: {self._lj_a.str_repr_short}\n"
            f"Lennard-Jones parameter B: {self._lj_b.str_repr_short}\n"
            f"{self._desc_bond_k_oh} (k_bond):\n"
            f"{self._bond_k_oh.str_repr_short} = {self._bond_k_oh_converted.str_repr_short}\n"
            f"{self._desc_bond_eq_len_oh} (r_OH):\n"
            f"{self._bond_eq_len_oh.str_repr_short} = {self._bond_eq_len_oh_converted.str_repr_short}\n"
            f"{self._desc_angle_k_hoh} (k_angle):\n"
            f"{self._angle_k_hoh.str_repr_short} = {self._angle_k_hoh_converted.str_repr_short}\n"
            f"{self._desc_angle_eq_hoh} (θ_HOH):\n"
            f"{self._angle_eq_hoh.str_repr_short} = {self._angle_eq_hoh_converted.str_repr_short}"
        )
        return str_repr

    @property
    def model_metadata(self):
        str_repr = (
            f"Name: {self.name}\n"
            f"Description: {self.description}\n"
            f"Reference: {self.publication_name}, {self.publication_citation} {self.publication_link}"
        )
        return str_repr

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def publication_name(self):
        return self._ref_name

    @property
    def publication_citation(self):
        return self._ref_cite

    @property
    def publication_link(self):
        return self._ref_link

    def open_publication_webpage(self):
        return webbrowser.open(self._ref_link)
