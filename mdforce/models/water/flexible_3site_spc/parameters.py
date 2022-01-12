

from pathlib import Path
import webbrowser


import pandas as pd
from duq.quantity import Quantity


class ModelParameters:

    def __init__(self, model_name):

        here = Path(__file__).parent
        df = pd.read_pickle(here/"params.pkl")
        self._dataframe = df

        self._name = model_name
        self._description = df.loc[model_name, ("Metadata", "Info", "Description")]
        self._ref_name = df.loc[model_name, ("Metadata", "Reference", "Name")]
        self._ref_cite = df.loc[model_name, ("Metadata", "Reference", "Citation")]
        self._ref_link = df.loc[model_name, ("Metadata", "Reference", "Link")]

        charge_o = df.loc[model_name, ("Parameters", "Coulomb", "q_O")].split()
        charge_h = df.loc[model_name, ("Parameters", "Coulomb", "q_H")].split()
        lj_epsilon_oo = df.loc[model_name, ("Parameters", "Lennard-Jones", "ε_OO")].split()
        lj_sigma_oo = df.loc[model_name, ("Parameters", "Lennard-Jones", "σ_OO")].split()
        bond_k_oh = df.loc[model_name, ("Parameters", "Bond vibration", "k")].split()
        bond_eq_len_oh = df.loc[model_name, ("Parameters", "Bond vibration", "r_OH")].split()
        angle_k_hoh = df.loc[model_name, ("Parameters", "Angle vibration", "k")].split()
        angle_eq_hoh = df.loc[model_name, ("Parameters", "Angle vibration", "θ_HOH")].split()

        self._charge_o = Quantity(float(charge_o[0]), charge_o[1])
        self._charge_h = Quantity(float(charge_h[0]), charge_h[1])
        self._lj_epsilon_oo = Quantity(float(lj_epsilon_oo[0]), lj_epsilon_oo[1])
        self._lj_sigma_oo = Quantity(float(lj_sigma_oo[0]), lj_sigma_oo[1])
        self._bond_k_oh = Quantity(float(bond_k_oh[0]), bond_k_oh[1])
        self._bond_eq_len_oh = Quantity(float(bond_eq_len_oh[0]), bond_eq_len_oh[1])
        self._angle_k_hoh = Quantity(float(angle_k_hoh[0]), angle_k_hoh[1])
        self._angle_eq_hoh = Quantity(float(angle_eq_hoh[0]), angle_eq_hoh[1])

    def unify_units(self, unit_length, unit_time, unit_mass="Da"):

        return

    @property
    def model_parameters(self):
        str_repr = ()
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

    @property
    def publication_webpage(self):
        webbrowser.open_new(self._ref_link)
        return
