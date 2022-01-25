"""
Implementation of the individual terms of a general force field.

Each function calculates the force for a given number of particle-pairs (or particle-triplets in
case of the angle-vibration function) in an iterative way. Moreover, each function also returns the
potential energy of each particle-pair/triplet.
"""

# Standard library
from typing import Tuple
# 3rd-party packages
import numpy as np
# Self
from . import terms_single_array


def coulomb(
        q: np.ndarray, pairs_idx: np.ndarray, c: np.ndarray, k_e: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the coulomb potential between a number of particle-pairs, and force on each particle.

    Parameters
    ----------
    q : numpy.ndarray
        Coordinates vectors of particles as a 2D-array of shape (n, m), where 'n' is the number of
        particles, and 'm' is the number of spatial dimensions for particles' coordinates.
    pairs_idx : numpy.ndarray
        Indices of all pairs of particles in `q`, between which the interaction should be
        calculated, as a 2D-array of shape (p, 2), where 'p' is the number of interacting pairs.
    c : numpy.ndarray
        Charges of all particles in `q`, as a 1D-array of shape (n, ). The value at each index
        corresponds to the charge of the particle at the same index in `q`.
    k_e : float
        Coulomb constant, i.e. (1 / 4πε0).

    Returns
    -------
    force, potential : Tuple[numpy.ndarray, numpy.ndarray]
        Force vectors for all particles, as a 2D-array of shape (n, m), followed by the potential
        energy between each pair, as a 1D-array of shape (p, ).

    Notes
    -----
    For particles in `q` whose index is not in `pairs_idx`, the respective positions in returned
    force and potential arrays will be zero.
    """
    # Initialize arrays to store force and potential
    f = np.zeros(q.shape)
    pot = np.zeros(pairs_idx.shape[0])
    # Iterate over all particle-pair indices, and calculate force and potential
    for pair_idx, (i, j) in enumerate(pairs_idx):
        f_ij, pot_ij = terms_single_array.coulomb(q[i], q[j], c[i], c[j], k_e)
        f[[i, j]] += f_ij, -f_ij
        pot[pair_idx] = pot_ij
    return f, pot


def lennard_jones(
        q: np.ndarray, pairs_idx: np.ndarray, a: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Lennard-Jones potential between a number of particle-pairs, and force on each
    particle.

    Parameters
    ----------
    q : numpy.ndarray
        Coordinates vectors of particles as a 2D-array of shape (n, m), where 'n' is the number of
        particles, and 'm' is the number of spatial dimensions for particles' coordinates.
    pairs_idx : numpy.ndarray
        Indices of all pairs of particles in `q`, between which the interaction should be
        calculated, as a 2D-array of shape (p, 2), where 'p' is the number of interacting pairs.
    a : numpy.ndarray
        A-parameter of the Lennard-Jones potential for each particle-pair in `pairs_idx`, as a
        1D-array of shape (p, ).
    b : numpy.ndarray
        B-parameter of the Lennard-Jones potential for each particle-pair in `pairs_idx`, as a
        1D-array of shape (p, ).

    Returns
    -------
    force, potential : Tuple[numpy.ndarray, numpy.ndarray]
        Force vectors for all particles, as a 2D-array of shape (n, m), followed by the potential
        energy between each pair, as a 1D-array of shape (p, ).

    Notes
    -----
    For particles in `q` whose index is not in `pairs_idx`, the respective positions in returned
    force and potential arrays will be zero.
    """
    # Initialize arrays to store force and potential
    f = np.zeros(q.shape)
    pot = np.zeros(pairs_idx.shape[0])
    # Iterate over all particle-pair indices, and calculate force and potential
    for pair_idx, (i, j) in enumerate(pairs_idx):
        f_ij, pot_ij = terms_single_array.lennard_jones(q[i], q[j], a[pair_idx], b[pair_idx])
        f[[i, j]] += f_ij, -f_ij
        pot[pair_idx] = pot_ij
    return f, pot


def bond_vibration_harmonic(
        q: np.ndarray, pairs_idx: np.ndarray, dist_eq: np.ndarray, k_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the harmonic bond vibration potential between a number of particle-pairs, and force
    on each particle.

    Parameters
    ----------
    q : numpy.ndarray
        Coordinates vectors of particles as a 2D-array of shape (n, m), where 'n' is the number of
        particles, and 'm' is the number of spatial dimensions for particles' coordinates.
    pairs_idx : numpy.ndarray
        Indices of all pairs of particles in `q`, between which the interaction should be
        calculated, as a 2D-array of shape (p, 2), where 'p' is the number of interacting pairs.
    dist_eq : numpy.ndarray
        Equilibrium bond length for each particle-pair in `pairs_idx`, as a 1D-array of shape
        (p, ).
    k_b : float
        Force constant of the harmonic bond potential for each particle-pair in `pairs_idx`, as a
        1D-array of shape (p, ).

    Returns
    -------
    force, potential : Tuple[numpy.ndarray, numpy.ndarray]
        Force vectors for all particles, as a 2D-array of shape (n, m), followed by the potential
        energy between each pair, as a 1D-array of shape (p, ).

    Notes
    -----
    For particles in `q` whose index is not in `pairs_idx`, the respective positions in returned
    force and potential arrays will be zero.
    """
    # Initialize arrays to store force and potential
    f = np.zeros(q.shape)
    pot = np.zeros(pairs_idx.shape[0])
    # Iterate over all particle-pair indices, and calculate force and potential
    for pair_idx, (i, j) in enumerate(pairs_idx):
        f_ij, pot_ij = terms_single_array.bond_vibration_harmonic(
            q[i], q[j], dist_eq[pair_idx], k_b[pair_idx]
        )
        f[[i, j]] += f_ij, -f_ij
        pot[pair_idx] = pot_ij
    return f, pot


def angle_vibration_harmonic(
        q: np.ndarray, triplets_idx: np.ndarray, angle_eq: np.ndarray, k_a: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the harmonic angle vibration potential between a number of particle-triplets, and
    force on each particle.

    Parameters
    ----------
    q : numpy.ndarray
        Coordinates vectors of particles as a 2D-array of shape (n, m), where 'n' is the number of
        particles, and 'm' is the number of spatial dimensions for particles' coordinates.
    triplets_idx : numpy.ndarray
        Indices of all triplets of particles in `q`, between which the interaction should be
        calculated, as a 2D-array of shape (p, 3), where 'p' is the number of interacting pairs.
        For each triplet, the index of the particle in the middle should be the first index.
    angle_eq : numpy.ndarray
        Equilibrium angle (in radian) for each particle-triplet in `triplets_idx`, as a 1D-array of
        shape (p, ).
    k_a : numpy.ndarray
        Force constant of the harmonic angle potential for each particle-triplet in `triplets_idx`,
        as a 1D-array of shape (p, ).

    Returns
    -------
    force, potential : Tuple[numpy.ndarray, numpy.ndarray]
        Force vectors for all particles, as a 2D-array of shape (n, m), followed by the potential
        energy between each triplet, as a 1D-array of shape (p, ).

    Notes
    -----
    For particles in `q` whose index is not in `pairs_idx`, the respective positions in returned
    force and potential arrays will be zero.
    """
    # Initialize arrays to store force and potential
    f = np.zeros(q.shape)
    pot = np.zeros(triplets_idx.shape[0])
    # Iterate over all particle-pair indices, and calculate force and potential
    for triplet_idx, (j, i, k) in enumerate(triplets_idx):
        f_j, f_i, f_k, pot_ijk = terms_single_array.angle_vibration_harmonic(
            q[j], q[i], q[k], angle_eq[triplet_idx], k_a[triplet_idx]
        )
        f[[j, i, k]] += f_j, f_i, f_k
        pot[triplet_idx] = pot_ijk
    return f, pot
