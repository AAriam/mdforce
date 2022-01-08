from pathlib import Path
from enum import Enum

import pandas as pd


def read_param_df(param_filepath):
    df = pd.read_csv(param_filepath)
    description = df.loc[0, "Description"]
    df.drop(0, inplace=True)
    df.set_index(["Force", "Parameter"], inplace=True)
    df.fillna(0, inplace=True)
    return df, description


def extract_params_from_df(param_df):
    data = {
        "LJ_A": param_df.loc[("Lennard-Jones", "A"), "Value"],
        "LJ_B": param_df.loc[("Lennard-Jones", "B"), "Value"],
        "COULOMB_K": param_df.loc[("Coulomb", "Coulomb constant"), "Value"],
        "CHARGE_H": param_df.loc[("Coulomb", "Charge Hydrogen"), "Value"],
        "CHARGE_O": param_df.loc[("Coulomb", "Charge Oxygen"), "Value"],
        "BOND_K": param_df.loc[("Bond vibration", "Force constant"), "Value"],
        "BOND_EQ_DIST": param_df.loc[("Bond vibration", "Equilibrium distance"), "Value"],
        "ANGLE_K": param_df.loc[("Angle vibration", "Force constant"), "Value"],
        "ANGLE_EQ_ANGLE": param_df.loc[("Angle vibration", "Equilibrium angle"), "Value"]
    }
    return data


def load_param_files():
    here = Path(__file__).parent
    param_files = list((here/"params").glob("*.csv"))
    param_data = {}
    for path in param_files:
        df, description = read_param_df(path)
        data = extract_params_from_df(df)
        param_data[path.stem] = {"description": description, "df": df, "data": data, "path": path}
    return param_data

param_data = load_param_files()

Parameters = Enum("Parameters", param_data)