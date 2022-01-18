"""
Module to construct the parameters dataframe for different flexible-3site-SPC water models.
This is only used once to create the dataframe `params.pkl`, but not needed afterwards,
unless other model parameters are to be added to the dataframe.
"""

# 3rd-party packages
import pandas as pd

# Create column names for the multi-index dataframe
columns = [
    ("Parameters", "Coulomb", "q_O"),
    ("Parameters", "Coulomb", "q_H"),
    ("Parameters", "Lennard-Jones", "ε_OO"),
    ("Parameters", "Lennard-Jones", "σ_OO"),
    ("Parameters", "Bond vibration", "k"),
    ("Parameters", "Bond vibration", "r_OH"),
    ("Parameters", "Angle vibration", "k"),
    ("Parameters", "Angle vibration", "θ_HOH"),
    ("Metadata", "Info", "Description"),
    ("Metadata", "Reference", "Name"),
    ("Metadata", "Reference", "Citation"),
    ("Metadata", "Reference", "Link"),
]

# Create the empty dataframe
column_indices = pd.MultiIndex.from_tuples(columns)
df = pd.DataFrame(columns=column_indices)
df.index.name = "Model"

# Write description for each column in the first row
df.loc["Description"] = [
    "Partial charge of the oxygen atom",
    "Partial charge of the hydrogen atom",
    "Lennard-Jones dispersion energy; i.e. depth of the oxygen–oxygen potential well",
    "Lennard-Jones size of the particle; i.e. oxygen–oxygen distance at which the potential is zero",
    "Force constant of the harmonic bond-vibration potential",
    "Equilibrium oxygen–hydrogen bond length",
    "Force constant of the harmonic angle-vibration potential",
    "Equilibrium hydrogen–oxygen–hydrogen bond angle",
    "Short description of the model",
    "Name of the publication",
    "Citation of the publication",
    "Link to the publication",
]


# Write parameters of different models in the next rows (one model per row)

df.loc["SPC/Fw"] = [
    "-0.82 e",
    "0.41 e",
    "0.1554253 kcal.mol^-1",
    "3.165492 Å",
    "1059.162 kcal.mol^-1.Å^-2",
    "1.012 Å",
    "75.90 kcal.mol^-1.rad^-2",
    "113.24 deg",
    "A new flexible three-site water model to better reflect dynamical and dielectric properties of bulk water.",
    "Flexible simple point-charge water model with improved liquid-state properties",
    "Y. Wu, H. L. Tepper, G. A. Voth, J. Chem. Phys. 2006, 124, 024503.",
    "https://doi.org/10.1063/1.2136877",
]


df.loc["SPC/Fd"] = [
    "-0.82 e",
    "0.41 e",
    "0.1554253 kcal.mol^-1",
    "3.165492 Å",
    "1054.20 kcal.mol^-1.Å^-2",
    "1.0 Å",
    "75.90 kcal.mol^-1.rad^-2",
    "109.5 deg",
    "",
    "Simple intramolecular model potentials for water",
    "L. X. Dang, B. M. Pettitt, J. Phys. Chem. 1987, 91, 12, 3349–3354.",
    "https://doi.org/10.1021/j100296a048",
]

df.loc["TIP3P/Fs"] = [
    "-0.834 e",
    "0.417 e",
    "0.1522 kcal.mol^-1",
    "3.1506 Å",
    "1054.20 kcal.mol^-1.Å^-2",
    "0.96 Å",
    "68.087 kcal.mol^-1.rad^-2",
    "104.5 deg",
    "",
    "The computer simulation of proton transport in water",
    "U. W. Schmitt, G. A. Voth, J. Chem. Phys. 1999, 111, 9361.",
    "https://doi.org/10.1063/1.480032",
]

# Save dataframe to pickle file
df.to_pickle("params.pkl")
