"""
Helper functions used by all models.
"""

# Standard library
from typing import Sequence, Union

# 3rd-party packages
import numpy as np
import duq


def raise_for_input_criteria(
    q: np.ndarray,
    mol_ids: np.ndarray,
    atomic_nums: np.ndarray,
    bonded_atoms_idxs: Sequence[np.ndarray],
):
    """
    Check the input data for any discrepancies and raise an Error if any is found.

    Parameters
    ----------
    q : numpy.ndarray
        2D-array of shape (n, m), containing the coordinates of 'n' atoms in an 'm'-dimensional space.
    mol_ids : numpy.ndarray
        1D-array of shape (n, ), containing the ID of the molecule, to which each atom in `q` belongs to.
        IDs should be integers, but the actual value of each ID does not play a role.
    atomic_nums : numpy.ndarray
        1D array of shape (n, ), containing the atomic number of each atom in `q`.
    bonded_atoms_idxs : Sequence[numpy.ndarray]
        2D-array of shape (n, variable), where each sub-array is a list of indices of atoms in `q`, to which
        the atom in `q` at the same index as the sub-array's index is bonded.

    Returns
    -------
        None

    Raises
    ------
    TypeError, ValueError
    """

    # Check input types
    if not isinstance(q, np.ndarray):
        raise TypeError("Type of `q` should be numpy.ndarray.")
    elif q.dtype.kind not in f"{np.typecodes['AllFloat']}{np.typecodes['AllInteger']}":
        raise TypeError("Data-type of `q` should be either float or integer.")
    elif not isinstance(mol_ids, np.ndarray):
        raise TypeError("Type of `mol_ids` should be numpy.ndarray.")
    elif mol_ids.dtype.kind not in np.typecodes["AllInteger"]:
        raise TypeError("Data-type of `mol_ids` should be integer.")
    elif not isinstance(atomic_nums, np.ndarray):
        raise TypeError("Type of `atomic_nums` should be numpy.ndarray.")
    elif atomic_nums.dtype.kind not in np.typecodes["AllInteger"]:
        raise TypeError("Data-type of `atomic_nums` should be integer.")
    elif not isinstance(bonded_atoms_idxs, (Sequence, np.ndarray)):
        raise TypeError("Type of `bonded_atoms_idxs` should be Sequence or numpy.ndarray.")
    else:
        for indices in bonded_atoms_idxs:
            if not isinstance(indices, np.ndarray):
                raise TypeError(
                    "Type of each element of array `bonded_atoms_idxs` should be numpy.ndarray."
                )
            elif indices.dtype.kind not in np.typecodes["AllInteger"]:
                raise TypeError("Sub-arrays in `bonded_atoms_idxs` should have an integer type.")
            else:
                pass

    # Check array dimensions
    if q.ndim != 2:
        raise ValueError("Array `q` should be 2-dimensional.")
    elif mol_ids.ndim != 1:
        raise ValueError("Array `mol_ids` should be 1-dimensional.")
    elif atomic_nums.ndim != 1:
        raise ValueError("Array `atomic_nums` should be 1-dimensional.")
    else:
        for indices in bonded_atoms_idxs:
            if indices.ndim != 1:
                raise ValueError("Each element of array `bonded_atoms_idxs` should be a 1D-array.")
            else:
                pass

    # Check completeness of data
    num_atoms = q.shape[0]
    if mol_ids.size != num_atoms:
        raise ValueError("Lengths of `mol_ids` and `q` do not match.")
    elif atomic_nums.size != num_atoms:
        raise ValueError("Lengths of `atomic_nums` and `q` do not match.")
    elif len(bonded_atoms_idxs) != num_atoms:
        raise ValueError("Lengths of `bonded_atoms_idxs` and `q` do not match.")

    return


def convert_to_unit(
    unit: Union[duq.Unit, str], correct_dimension: str, param_name: str
) -> duq.Unit:
    """
    Verify that a given `unit` is either a string or a `duq.Unit` object, and raise an error
    otherwise. If it's a string, transform it to a `duq.Unit` object. Verify that the `duq.Unit`
    object has the expected dimension, and raise an error otherwise.

    Parameters
    ----------
    unit : Union[duq.Unit, str]
        Unit of interest, either as a string representation (see duq.Unit for more details) or
        a `duq.Unit` object.
    correct_dimension : str
        String representation of the expected dimension of the unit (see duq.Dimension for more
        details).
    param_name : str
        Name of the parameter to which `unit` is bound; to be mentioned in the error message.

    Returns
    -------
        duq.Unit
        Object representing the unit of interest.

    Raises
    ------
    ValueError
    """
    if isinstance(unit, str):
        unit_obj = duq.Unit(unit)
    elif isinstance(unit, duq.Unit):
        unit_obj = unit
    else:
        raise ValueError(f"Type of parameter `{param_name}` should be either duq.Unit or string.")
    raise_for_dimension(unit_obj, correct_dimension, param_name)
    return unit_obj


def convert_to_quantity(
    quantity: Union[duq.Quantity, str], correct_dimension: str, param_name: str
) -> duq.Quantity:
    """
    Verify that a given `quantity` is either a string or a `duq.Quantity` object, and raise an
    error otherwise. If it's a string, transform it to a `duq.Quantity` object. Verify that the
    `duq.Quantity` object has the expected dimension, and raise an error otherwise.

    Parameters
    ----------
    quantity : Union[duq.Quantity, str]
        Quantity of interest, either as a string representation (see duq.Quantity for more details)
        or a `duq.Quantity` object.
    correct_dimension : str
        String representation of the expected dimension of the quantity (see duq.Dimension for more
        details).
    param_name : str
        Name of the parameter to which `quantity` is bound; to be mentioned in the error message.

    Returns
    -------
        duq.Quantity
        Object representing the quantity of interest.

    Raises
    ------
    ValueError
    """
    if isinstance(quantity, str):
        quantity_value, quantity_unit = quantity.split()
        quantity_obj = duq.Quantity(float(quantity_value), quantity_unit)
    elif isinstance(quantity, duq.Quantity):
        quantity_obj = quantity
    else:
        raise ValueError(
            f"Type of parameter `{param_name}` should be either duq.Quantity or string."
        )
    raise_for_dimension(quantity_obj, correct_dimension, param_name)
    return quantity_obj


def raise_for_dimension(
    unit_or_quantity: Union[duq.Unit, duq.Quantity],
    correct_dimension: Union[str, duq.Dimension],
    param_name: str,
) -> None:
    """
    Verify that a `duq.Quantity` or `duq.Unit` object has the correct dimension,
    and raise an error otherwise.

    Parameters
    ----------
    unit_or_quantity : Union[duq.Unit, duq.Quantity]
        Object whose dimension is to be verified.
    correct_dimension : Union[str, duq.Dimension]
        String representation or duq.Dimension object of the expected dimension
        (see duq.Dimension for more details).
    param_name : str
        Name of the parameter to which the `duq.Unit` or `duq.Quantity` object is bound; to be
        mentioned in the error message.

    Returns
    -------
        None

    Raises
    ------
    ValueError
    """
    if isinstance(correct_dimension, str):
        correct_dimension = duq.Dimension(correct_dimension)
    if not unit_or_quantity.dimension.is_convertible_to(correct_dimension):
        raise ValueError(
            f"Parameter `{param_name}` should have the physical dimension of "
            f"{correct_dimension.symbol_as_is}, but has {unit_or_quantity.dimension.symbol_as_is}."
        )
    return
