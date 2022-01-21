"""
Helper functions used by all models.
"""

# Standard library
from typing import Sequence

# 3rd-party packages
import numpy as np


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
