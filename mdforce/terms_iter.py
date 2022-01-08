import numpy as np

from . import terms


def lennard_jones(q, pair_idx, ab):
    """
    Calculate the total Lennard-Jones potential and force
    for a number of particles, due to other particles.

    Parameters
    ----------
    q : numpy.ndarray
        2D array of shape (n, m), containing the coordinates
        of n particles in an m-dimensional space.
    pair_idx : numpy.ndarray
        2D array of shape (p, 2), containing the indices
        of all pairs of particles in q, between which
        the Lennard-Jones interaction should be calculated.
    ab : numpy.ndarray
        2D array of shape (p, 2), containing the parameters
        A and B , respectively, for each pair in pair_idx.

    Returns
    -------
        tuple [numpy.ndarray, numpy.ndarray]
        Force vectors for all particles, as a 2D array of shape (n, m)
        followed by potential energy of all particles, as a 1D array of length (n).

    Notes
    -----
    For particles in q whose index is not in pair_idx,
    the respective positions in returned force and potential arrays
    will be zero.
    """

    f = np.zeros_like(q)
    e = np.zeros(q.shape[0])
    for idx, (i, j) in enumerate(pair_idx):
        f_ij, e_ij = terms.lennard_jones(q[i], q[j], ab[idx, 0], ab[idx, 1])
        f[[i, j]] += f_ij, -f_ij
        e[[i, j]] += e_ij
    return f, e


def coulomb(q, pair_idx, c, k):
    """
    Calculate the total Coulomb potential and force
    for a number of particles, due to other particles.

    Parameters
    ----------
    q : numpy.ndarray
        2D array of shape (n, m), containing the coordinates
        of n particles in an m-dimensional space.
    pair_idx : numpy.ndarray
        2D array of shape (p, 2), containing the indices
        of all pairs of particles in q, between which
        the Coulomb interaction should be calculated.
    c : numpy.ndarray
        1D array of length (n), containing the
        charges of all particles in q.
    k : float
        Coulomb constant, i.e. (1 / 4πε0).

    Returns
    -------
        tuple [numpy.ndarray, numpy.ndarray]
        Force vectors for all particles, as a 2D array of shape (n, m)
        followed by potential energy of all particles, as a 1D array of length (n).

    Notes
    -----
    For particles in q whose index is not in pair_idx,
    the respective positions in returned force and potential arrays
    will be zero.
    """

    f = np.zeros_like(q)
    e = np.zeros(q.shape[0])
    for i, j in pair_idx:
        f_ij, e_ij = terms.coulomb(q[i], q[j], c[i], c[j], k)
        f[[i, j]] += f_ij, -f_ij
        e[[i, j]] += e_ij
    return f, e


def bond_vibration_harmonic(q, pair_idx, eq_dist, k):
    """
    Calculate the total bond-vibration potential and force
    for a number of bonded pairs of particles, using the
    harmonic oscillator model.

    Parameters
    ----------
    q : numpy.ndarray
        2D array of shape (n, m), containing the coordinates
        of n particles in an m-dimensional space.
    pair_idx : numpy.ndarray
        2D array of shape (p, 2), containing the indices
        of all pairs of particles in q, between which
        the bond-vibration interaction should be calculated.
    eq_dist : numpy.ndarray
        1D array of length (p), containing the
        equilibrium bond length for each pair in pair_idx.
    k : float
        force constant of the harmonic potential.

    Returns
    -------
        tuple [numpy.ndarray, numpy.ndarray]
        Force vectors for all particles, as a 2D array of shape (n, m)
        followed by potential energy of all particles, as a 1D array of length (n).

    Notes
    -----
    For particles in q whose index is not in pair_idx,
    the respective positions in returned force and potential arrays
    will be zero.
    """

    f = np.zeros_like(q)
    e = np.zeros(q.shape[0])
    for idx, (i, j) in enumerate(pair_idx):
        f_ij, e_ij = terms.bond_vibration_harmonic(q[i], q[j], eq_dist[idx], k)
        f[[i, j]] += f_ij, -f_ij
        e[[i, j]] += e_ij
    return f, e


def angle_vibration_harmonic(q, pair_idx, eq_angle, k):
    """
    Calculate the total angle-vibration potential and force
    for a number of bonded triplets of particles, using the
    harmonic oscillator model.

    Parameters
    ----------
    q : numpy.ndarray
        2D array of shape (n, m), containing the coordinates
        of n particles in an m-dimensional space.
    pair_idx : numpy.ndarray
        2D array of shape (p, 3), containing the indices
        of all triplets of particles in q, between which
        the angle-vibration interaction should be calculated.
        For each triplet, the index of the particle in the middle
        should be the first index
    eq_angle : numpy.ndarray
        1D array of length (p), containing the
        equilibrium angle for each triplet in pair_idx.
    k : float
        force constant of the harmonic potential.

    Returns
    -------
        tuple [numpy.ndarray, numpy.ndarray]
        Force vectors for all particles, as a 2D array of shape (n, m)
        followed by potential energy of all particles, as a 1D array of length (n).

    Notes
    -----
    For particles in q whose index is not in pair_idx,
    the respective positions in returned force and potential arrays
    will be zero.
    """

    f = np.zeros_like(q)
    e = np.zeros(q.shape[0])
    # FIXME
    # remove angle (added for debugging)
    for idx, (m, l, r) in enumerate(pair_idx):
        f_m, f_l, f_r, e_mlr, angle = terms.angle_vibration(q[m], q[l], q[r], eq_angle[idx], k)
        f[[m, l, r]] += f_m, f_l, f_r
        e[[m, l, r]] += e_mlr

    return f, e, angle
