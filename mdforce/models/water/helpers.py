"""
Helper functions used by all water models.
"""

# Standard library
from typing import Sequence

# 3rd-party packages
import numpy as np

# Self
from ...helpers import raise_for_input_criteria


def check_input_data(
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
        IDs should be integers, and each ID should appear 3 times in the array, corresponding to 2 hydrogens
        and one oxygen atom that comprise that molecule. Otherwise, the actual value of each ID does not play a role.
    atomic_nums : numpy.ndarray
        1D array of shape (n, ), containing the atomic number of each atom in `q`.
        Since this is a water model, the only acceptable values are 1 and 8.
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

    # Check general criteria
    raise_for_input_criteria(q, mol_ids, atomic_nums, bonded_atoms_idxs)

    unique_atom_types, count_atom_types = np.unique(atomic_nums, return_counts=True)
    num_atoms = q.shape[0]
    # Verify that there are 3N atom coordinates
    if num_atoms % 3 != 0:
        raise ValueError("Number of atoms should be a multiple of 3.")
    # Verify that there are only hydrogen and oxygen
    elif np.all(unique_atom_types != [1, 8]):
        raise ValueError("Only oxygen and hydrogen atoms are allowed.")
    # Verify that number of hydrogen atoms is two times the number of oxygen atoms
    elif count_atom_types[0] != count_atom_types[1] * 2:
        raise ValueError("Number of hydrogen atoms should be two times the number of oxygen atoms.")
    else:
        for i, indices in enumerate(bonded_atoms_idxs):
            # Verify that each atom is connected to either one or two atoms
            if indices.size not in [1, 2]:
                raise ValueError("Each element of array `bonded_atoms_idxs` should have either 1 or 2 elements.")
            elif indices.size == 1:
                # Verify that atoms with one bond are hydrogen atoms
                if atomic_nums[i] == 8:
                    raise ValueError(f"Oxygen atom at index {i} is connected to only one atom.")
                # Verify that hydrogen atoms are connected to oxygen atoms
                elif atomic_nums[indices[0]] != 8:
                    raise ValueError(f"Hydrogen atom at index {i} is connected to another hydrogen atom.")
                else:
                    pass
            elif indices.size == 2:
                # Verify that atoms with two bonds are oxygen atoms
                if atomic_nums[i] == 1:
                    raise ValueError(f"Hydrogen atom at index {i} is connected to two atoms.")
                # Verify that oxygen atoms are connected to two hydrogen atoms
                elif atomic_nums[indices[0]] != 1 or atomic_nums[indices[1]] != 1:
                    raise ValueError(f"Oxygen atom at index {i} is connected to another oxygen atom.")
                else:
                    pass
            else:
                pass

    return
