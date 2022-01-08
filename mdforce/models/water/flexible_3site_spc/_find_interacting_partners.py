import numpy as np


def find_interacting_partners_indices(mol_ids, atom_ids, bonded_atoms_idx):
    """
    Find all unique interacting partners for each type of interaction
    according to the model, and return their indices.

    Parameters
    ----------
    mol_ids : numpy.ndarray
        1D array of length `n` (n = number of atoms) containing the
        molecule-IDs of all atoms. They can have arbitrary values,
        but should have the same value for all atoms in the same molecule.
    atom_ids : numpy.ndarray
        1D array of length `n` containing the atom-IDs (int) of all atoms.
        Atom-Ids should be the atomic number of each atom.
        Therefore, here, only the value 1 (for hydrogen) and 8 (for oxygen) are accepted.
    bonded_atoms_idx : list
        2D list of length `n` containing the indices (int) of all bonded atoms to each atom.

    Returns
    -------
        tuple
        Tuple of four 2D numpy arrays, containing the indices of all unique
        interacting partners for each interaction type, in the order:
        (Lennard-Jones, Coulomb, Bond-vibration, Angle-vibration)
    """
    check_input_data(mol_ids, atom_ids, bonded_atoms_idx)
    idx_lennard_jones = lennard_jones(atom_ids)
    idx_coulomb = coulomb(mol_ids)
    idx_bond_vibration = bond_vibration(bonded_atoms_idx)
    idx_angle_vibration = angle_vibration(bonded_atoms_idx)
    return idx_lennard_jones, idx_coulomb, idx_bond_vibration, idx_angle_vibration


def check_input_data(mol_ids, atom_ids, bonded_atoms_idx):

    num_atoms = atom_ids.shape[0]
    unique_atom_types, count_atom_types = np.unique(atom_ids, return_counts=True)

    # Verify that all arrays have the same length
    if (
            num_atoms != mol_ids.size
    ) or (
            num_atoms != len(bonded_atoms_idx)
    ):
        raise ValueError("All input arrays should have the same length.")

    # Verify that there are 3N atom coordinates
    elif num_atoms % 3 != 0:
        raise ValueError("Number of atoms should be a multiple of 3.")

    # Verify that there are only hydrogen and oxygen
    elif np.all(unique_atom_types != [1, 8]):
        raise ValueError("Only oxygen and hydrogen atoms are allowed.")

    # Verify that number of hydrogen atoms is two times the number of oxygen atoms
    elif count_atom_types[0] != count_atom_types[1] * 2:
        raise ValueError("Number of hydrogen atoms should be two times the number of oxygen atoms.")

    else:
        pass
    return


def calculate_sorted_indices(atom_ids, mol_ids):
    """
    Create an array of atom indices, where the atoms are first sorted
    by their molecule-ID, and then by their atom type.

    Parameters
    ----------
    atom_ids : numpy.ndarray
        1D array containing the type of each atom in the input data.
    mol_ids : numpy.ndarray
        1D array containing the molecule-ID of each atom in the input data.

    Returns
    -------
        numpy.ndarray
        Index array of all atoms in the input data,
        first sorted by their molecule-ID, and then
        sorted by their atom type.

    Examples
    --------
    Let's say A_n_m denotes the nth A atom in mth molecule;
    then an input data `q` = [H_1_1, H_1_2, H_2_1, H_2_2, O_1_1, O_1_2],
    will have `atom_types` = [1, 1, 1, 1, 8, 8]
    and `mol_ids` = [1, 2, 1, 2, 1, 2].
    Applying this function to `atom_types` and `mol_ids` will then return:
    [0, 2, 4, 1, 3, 5]
    Therefore, applying this index array to `q` will return:
    [H_1_1, H_2_1, O_1_1, H_1_2, H_2_2, O_1_2]
    """
    # Create array of atom indices
    atom_idx = np.arange(atom_ids.size)
    # Calculate new indices when atoms are sorted by their molecule ID
    atom_idx_sorted_by_mol_id = mol_ids.argsort()
    # Update atom indices
    atom_idx = atom_idx[atom_idx_sorted_by_mol_id]
    # Get array of atom types based on sorting
    atom_types_sorted = atom_ids[atom_idx_sorted_by_mol_id]
    # Calculate new indices when atoms are again sorted, now by their atom type
    for n in range(0, atom_types_sorted.shape[0] - 2, 3):
        atom_types_sorted_idx = atom_types_sorted[n:n + 3].argsort() + n
        atom_idx[n:n + 3][...] = atom_idx[atom_types_sorted_idx]
    return atom_idx


def calculate_pair_indices_coulomb_(atom_idx):
    """
    Create an array containing the indices of all unique pairs of atoms,
    for which the Coulomb interaction should be calculated, i.e. all unique
    pairs of atoms that are not in the same molecule.

    Parameters
    ----------
    atom_idx : numpy.ndarray
        1D array containing the indices of all atoms, sorted by molecule-ID,
        and within each molecule, by the order H, H, O (i.e. output of `calculate_sorted_indices`).

    Returns
    -------
        numpy.ndarray
        2D array containing the indices of all unique pairs of atoms,
        for which the Coulomb interaction should be calculated.
    """

    # Calculate the total number of Coulomb interactions that should be calculated;
    # Interactions are calculated between all unique atom pairs, as long as they are
    # not in the same water molecule. Therefore, for a unique pair of water molecules,
    # 9 interactions should be calculated. The number of unique pairs of water molecules
    # is then equal to the binomial coefficient (num_mols, 2) = num_mols * (num_mols - 1) / 2.
    # Therefore, the number of total interactions is equal to:
    num_mols = atom_idx.size // 3
    num_interactions = 9 * num_mols * (num_mols - 1) // 2
    # Initialize array to store atom indices of each unique pair of atoms, for which
    # the Coulomb interaction should be calculated
    pairs_idx = np.zeros((num_interactions, 2), dtype=np.intc)

    # Create counter for index of interaction, for storing the index pair in `pairs_idx`
    interaction_idx = 0
    # Iterate over all atom indices
    for idx_elem1, idx_atom1 in enumerate(atom_idx):
        # For each atom, the index of first interacting partner is the index of
        # first atom in the next molecule
        idx_first_partner = idx_elem1 + 3 - idx_elem1 % 3
        # Iterate over all interacting partners
        for idx_atom2 in atom_idx[idx_first_partner:]:
            # Assign atom index pairs of an interacting pair to `pairs_idx`
            # at position `interaction_idx`
            pairs_idx[interaction_idx][...] = [idx_atom1, idx_atom2]
            # increment `interaction_idx` by 1
            interaction_idx += 1
    return pairs_idx


def coulomb(mol_ids):
    """
    Create an array containing the indices of all unique pairs of atoms,
    for which the Coulomb interaction should be calculated, i.e. all unique
    pairs of atoms that are not in the same molecule.

    Parameters
    ----------
    mol_ids : numpy.ndarray
        1D array containing the molecule-IDs of all atoms.

    Returns
    -------
        numpy.ndarray
        2D array containing the indices of all unique pairs of atoms,
        for which the Coulomb interaction should be calculated.
    """

    # Calculate the total number of Coulomb interactions that should be calculated;
    # Interactions are calculated between all unique atom pairs, as long as they are
    # not the same atom and are not in the same water molecule. Therefore, for each atom,
    # (n - 3) interactions should be calculated (where `n` is the total number of atoms).
    # Therefore, the total number of interactions is equal to (n * (n - 3)).
    # Here, each pair is counted twice, thus the total number of unique pairs is (n * (n - 3) / 2).
    num_atoms = mol_ids.size
    num_interactions = num_atoms * (num_atoms - 3) // 2
    # Initialize array to store atom indices of each unique pair of atoms, for which
    # the Coulomb interaction should be calculated
    pairs_idx = np.zeros((num_interactions, 2), dtype=np.intc)

    # Store the index of first empty entry in `pairs_idx`
    first_empty_idx = 0
    # Iterate over all atoms
    for idx, mol_id in enumerate(mol_ids):
        # Create a boolean masking array for each atom,
        # based on the condition that the interacting partners
        # should not be in the same molecule.
        mask_different_mol = mol_ids != mol_id
        # Get the index of all atoms that are not in the same molecule as the current atom
        partners_idx = np.where(mask_different_mol)[0]
        # Create a boolean masking array for all atoms that are not in the same molecule,
        # based on the condition that they should have a higher index than the index of current atom.
        # This is so that every unique interaction is counted once and not twice.
        mask_unique_pairs = partners_idx > idx
        # Filter the index of partners to contain only those with higher index
        partners_idx_unique = partners_idx[mask_unique_pairs]
        # Assign the index of current atom to `pairs_idx`
        pairs_idx[first_empty_idx: first_empty_idx + partners_idx_unique.size, 0] = idx
        # Assign all indices of interacting partners to `pairs_idx`
        pairs_idx[first_empty_idx:first_empty_idx + partners_idx_unique.size, 1][...] = partners_idx_unique
        # Update the index of first empty entry in `pairs_idx`
        first_empty_idx += partners_idx_unique.size
    return pairs_idx


def lennard_jones(atom_ids):
    """
    Create an array containing the indices of all unique pairs of atoms,
    for which the Lennard–Jones interaction should be calculated, i.e.
    all unique pairs of oxygen atoms.

    Parameters
    ----------
    atom_ids : numpy.ndarray
        1D array containing the atom-IDs of all atoms.

    Returns
    -------
        numpy.ndarray
        2D array containing the indices of all unique pairs of atoms,
        for which the Lennard–Jones interaction should be calculated.
    """

    # Get the index of all oxygen atoms
    oxygens_idx = np.where(atom_ids == 8)[0]
    # Calculate the total number of Lennard–Jones interactions that should be calculated;
    # Interactions are calculated between all unique pairs of oxygen atoms. Therefore, when `n`
    # is the number of all oxygen atoms, the total number of unique interactions to be calculated
    # is equal to the binomial coefficient (n, 2) = n * (n - 1) / 2
    num_oxygen_atoms = oxygens_idx.size
    num_interactions = num_oxygen_atoms * (num_oxygen_atoms - 1) // 2
    # Initialize array to store atom indices of each unique pair of atoms, for which
    # the Lennard–Jones interaction should be calculated
    pairs_idx = np.zeros((num_interactions, 2), dtype=np.intc)

    # Store the index of first empty entry in `pairs_idx`
    first_empty_idx = 0
    # Iterate over all oxygen indices
    for elem_idx, oxygen_idx in enumerate(oxygens_idx):
        # Take the remaining of the array as partners indices
        partners_idx = oxygens_idx[elem_idx+1:]
        # Assign the index of current oxygen to `pairs_idx`
        pairs_idx[first_empty_idx:first_empty_idx + partners_idx.size, 0] = oxygen_idx
        # Assign all indices of interacting partners to `pairs_idx`
        pairs_idx[first_empty_idx:first_empty_idx + partners_idx.size, 1][...] = partners_idx
        # Update the index of first empty entry in `pairs_idx`
        first_empty_idx += partners_idx.size
    return pairs_idx


def bond_vibration(bonded_atoms_idx):
    """
    Create an array containing the indices of all unique pairs of atoms,
    for which the bond vibration interaction should be calculated, i.e.
    all bonded atom pairs.

    Parameters
    ----------
    bonded_atoms_idx : list
        2D list of ints containing the indices of all bonded atoms to each atom.

    Returns
    -------
        numpy.ndarray
        2D array containing the indices of all unique pairs of atoms,
        for which the bond vibration interaction should be calculated.
    """
    # Calculate the total number of bond vibration interactions that should be calculated;
    # Interactions are calculated between all unique bonded pairs of atoms. This is simply
    # equal to the size of the `bonded_atoms_idx` array divided by 2 (since each bond is noted twice).
    num_interactions = sum(map(len, bonded_atoms_idx)) // 2
    # Initialize array to store atom indices of each unique pair of atoms, for which
    # the bond vibration interaction should be calculated
    pairs_idx = np.zeros((num_interactions, 2), dtype=np.intc)

    # Store the index of first empty entry in `pairs_idx`
    first_empty_idx = 0
    # Iterate over list of bonded atoms indices for each atom
    for atom_idx, bonded_atoms_idx_list in enumerate(bonded_atoms_idx):
        # Turn each sub-list into a numpy array
        bonded_atoms_idx_list = np.array(bonded_atoms_idx_list)
        # Create a boolean masking array for each atom,
        # based on the condition that the index of bonded atom
        # should be greater than the index of current atom (to avoid counting each bonded pair twice).
        mask_unique_pairs = bonded_atoms_idx_list > atom_idx
        # Filter the index of bonded partners to contain only those with higher index
        partners_idx_unique = bonded_atoms_idx_list[mask_unique_pairs]
        # Assign the index of current atom to `pairs_idx`
        pairs_idx[first_empty_idx:first_empty_idx + partners_idx_unique.size, 0] = atom_idx
        # Assign all indices of bonded partners to `pairs_idx`
        pairs_idx[first_empty_idx:first_empty_idx + partners_idx_unique.size, 1][...] = partners_idx_unique
        # Update the index of first empty entry in `pairs_idx`
        first_empty_idx += partners_idx_unique.size
    return pairs_idx


def angle_vibration(bonded_atoms_idx):
    """
    Create an array containing the indices of all unique triplets atoms,
    for which the angle vibration interaction should be calculated, i.e.
    all unique bonded atom triplets.

    Parameters
    ----------
    bonded_atoms_idx : list
        2D list of ints containing the indices of all bonded atoms to each atom.

    Returns
    -------
        numpy.ndarray
        2D array containing the indices of all unique pairs of atoms,
        for which the angle vibration interaction should be calculated.
    """
    # Calculate the total number of bond vibration interactions that should be calculated;
    # Interactions are calculated between all unique bonded triplet of atoms. This is simply
    # equal to the size of the `bonded_atoms_idx` array divided by 4 (since each bond is noted twice,
    # and each two bonds make a bonded triplet).
    num_interactions = sum(map(len, bonded_atoms_idx)) // 4
    # Initialize array to store atom indices of each unique pair of atoms, for which
    # the bond vibration interaction should be calculated
    pairs_idx = np.zeros((num_interactions, 3), dtype=np.intc)

    # Store the index of first empty entry in `pairs_idx`
    first_empty_idx = 0
    # Iterate over list of bonded atoms indices for each atom
    for atom_idx, bonded_atoms_idx_list in enumerate(bonded_atoms_idx):
        # Look for atoms with more than 1 bonded atom; these are the oxygen atoms
        if len(bonded_atoms_idx_list) < 2:
            continue
        else:
            bonded_atoms_idx_list = np.array(bonded_atoms_idx_list)
            pairs_idx[first_empty_idx][0] = atom_idx
            pairs_idx[first_empty_idx][1:][...] = bonded_atoms_idx_list
            first_empty_idx += 1
    return pairs_idx


