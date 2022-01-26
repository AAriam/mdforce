"""
Test functions for individual force-field terms in `mdforce.terms_multi_iter`, evaluated using
functions in `mdforce.terms_single_array_simplified`.
"""


# 3rd-party packages
import numpy as np

# Self
from mdforce import terms_single_array_simplified as terms_single
from mdforce import terms_multi_iter as terms_iter


# Set up random number generator with seed to make sure testing results are consistent
random_gen = np.random.RandomState(1111)


def test_coulomb():
    """
    Test function for `mdforce.terms_multi_iter.coulomb`.
    """
    # Create an index array of pairs, for which the interaction should be calculated
    pairs_idx = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3]])
    # Run the test 100 times
    for i in range(100):
        # Create random x-y-z-coordinates for 5 particles
        q = random_gen.random_sample((5, 3)) * 2 - 1
        # Create random charges for 5 particles
        c = random_gen.random_sample(5) * 2 - 1
        # Create random force constant
        k_e = random_gen.random_sample() * 2 - 1
        # Calculate force and potential between each pair in `pairs_idx`, using the already tested
        # function in `terms_single_array_simplified`
        f01, e01 = terms_single.coulomb(q_i=q[0], q_j=q[1], c_i=c[0], c_j=c[1], k_e=k_e)
        f02, e02 = terms_single.coulomb(q_i=q[0], q_j=q[2], c_i=c[0], c_j=c[2], k_e=k_e)
        f03, e03 = terms_single.coulomb(q_i=q[0], q_j=q[3], c_i=c[0], c_j=c[3], k_e=k_e)
        f04, e04 = terms_single.coulomb(q_i=q[0], q_j=q[4], c_i=c[0], c_j=c[4], k_e=k_e)
        f12, e12 = terms_single.coulomb(q_i=q[1], q_j=q[2], c_i=c[1], c_j=c[2], k_e=k_e)
        f13, e13 = terms_single.coulomb(q_i=q[1], q_j=q[3], c_i=c[1], c_j=c[3], k_e=k_e)
        # Calculate force and potential between each pair in `pairs_idx`, using the equivalent
        # function in `terms_multi_iter`
        f_calc, e_calc = terms_iter.coulomb(q=q, pairs_idx=pairs_idx, c=c, k_e=k_e)
        # Verify that both functions' results are the same
        assert np.all(f_calc[0] == f01 + f02 + f03 + f04)
        assert np.all(f_calc[1] == -f01 + f12 + f13)
        assert np.all(f_calc[2] == -f02 - f12)
        assert np.all(f_calc[3] == -f03 - f13)
        assert np.all(f_calc[4] == -f04)
        assert e_calc[0] == e01
        assert e_calc[1] == e02
        assert e_calc[2] == e03
        assert e_calc[3] == e04
        assert e_calc[4] == e12
        assert e_calc[5] == e13
    return


def test_lennard_jones():
    """
    Test function for `mdforce.terms_multi_iter.lennard_jones`.
    """
    # Create an index array of pairs, for which the interaction should be calculated
    pairs_idx = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3]])
    # Run the test 100 times
    for i in range(100):
        # Create random x-y-z-coordinates for 5 particles
        q = random_gen.random_sample((5, 3)) * 2 - 1
        # Create random A-parameters for the 6 interacting pairs
        a = random_gen.random_sample(6) * 2 - 1
        # Create random B-parameters for the 6 interacting pairs
        b = random_gen.random_sample(6) * 2 - 1
        # Calculate force and potential between each pair in `pairs_idx`, using the already tested
        # function in `terms_single_array_simplified`
        f01, e01 = terms_single.lennard_jones(q_i=q[0], q_j=q[1], a_ij=a[0], b_ij=b[0])
        f02, e02 = terms_single.lennard_jones(q_i=q[0], q_j=q[2], a_ij=a[1], b_ij=b[1])
        f03, e03 = terms_single.lennard_jones(q_i=q[0], q_j=q[3], a_ij=a[2], b_ij=b[2])
        f04, e04 = terms_single.lennard_jones(q_i=q[0], q_j=q[4], a_ij=a[3], b_ij=b[3])
        f12, e12 = terms_single.lennard_jones(q_i=q[1], q_j=q[2], a_ij=a[4], b_ij=b[4])
        f13, e13 = terms_single.lennard_jones(q_i=q[1], q_j=q[3], a_ij=a[5], b_ij=b[5])
        # Calculate force and potential between each pair in `pairs_idx`, using the equivalent
        # function in `terms_multi_iter`
        f_calc, e_calc = terms_iter.lennard_jones(q=q, pairs_idx=pairs_idx, a=a, b=b)
        # Verify that both functions' results are the same
        assert np.all(f_calc[0] == f01 + f02 + f03 + f04)
        assert np.all(f_calc[1] == -f01 + f12 + f13)
        assert np.all(f_calc[2] == -f02 - f12)
        assert np.all(f_calc[3] == -f03 - f13)
        assert np.all(f_calc[4] == -f04)
        assert e_calc[0] == e01
        assert e_calc[1] == e02
        assert e_calc[2] == e03
        assert e_calc[3] == e04
        assert e_calc[4] == e12
        assert e_calc[5] == e13
    return


def test_bond_vibration_harmonic():
    """
    Test function for `mdforce.terms_multi_iter.bond_vibration_harmonic`.
    """
    # Create an index array of pairs, for which the interaction should be calculated
    pairs_idx = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3]])
    # Run the test 100 times
    for i in range(100):
        # Create random x-y-z-coordinates for 5 particles
        q = random_gen.random_sample((5, 3)) * 2 - 1
        # Create random equilibrium bond distances for the 6 interacting pairs
        d0 = random_gen.random_sample(6) * 2 - 1
        # Create random force constants for the 6 interacting pairs
        k_b = random_gen.random_sample(6) * 2 - 1
        # Calculate force and potential between each pair in `pairs_idx`, using the already tested
        # function in `terms_single_array_simplified`
        f01, e01 = terms_single.bond_vibration_harmonic(q_i=q[0], q_j=q[1], d0=d0[0], k_b=k_b[0])
        f02, e02 = terms_single.bond_vibration_harmonic(q_i=q[0], q_j=q[2], d0=d0[1], k_b=k_b[1])
        f03, e03 = terms_single.bond_vibration_harmonic(q_i=q[0], q_j=q[3], d0=d0[2], k_b=k_b[2])
        f04, e04 = terms_single.bond_vibration_harmonic(q_i=q[0], q_j=q[4], d0=d0[3], k_b=k_b[3])
        f12, e12 = terms_single.bond_vibration_harmonic(q_i=q[1], q_j=q[2], d0=d0[4], k_b=k_b[4])
        f13, e13 = terms_single.bond_vibration_harmonic(q_i=q[1], q_j=q[3], d0=d0[5], k_b=k_b[5])
        # Calculate force and potential between each pair in `pairs_idx`, using the equivalent
        # function in `terms_multi_iter`
        f_calc, e_calc = terms_iter.bond_vibration_harmonic(
            q=q, pairs_idx=pairs_idx, d0=d0, k_b=k_b
        )
        # Verify that both functions' results are the same
        assert np.all(f_calc[0] == f01 + f02 + f03 + f04)
        assert np.all(f_calc[1] == -f01 + f12 + f13)
        assert np.all(f_calc[2] == -f02 - f12)
        assert np.all(f_calc[3] == -f03 - f13)
        assert np.all(f_calc[4] == -f04)
        assert e_calc[0] == e01
        assert e_calc[1] == e02
        assert e_calc[2] == e03
        assert e_calc[3] == e04
        assert e_calc[4] == e12
        assert e_calc[5] == e13
    return


def test_angle_vibration_harmonic():
    """
    Test function for `mdforce.terms_multi_iter.angle_vibration_harmonic`.
    """
    # Create an index array of triplets, for which the interaction should be calculated
    triplets_idx = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 2], [1, 2, 3], [1, 3, 4]])
    # Run the test 100 times
    for i in range(100):
        # Create random x-y-z-coordinates for 5 particles
        q = random_gen.random_sample((5, 3)) * 2 - 1
        # Create random equilibrium bond distances for the 6 interacting pairs
        angle0 = random_gen.random_sample(6) * 2 - 1
        # Create random force constants for the 6 interacting pairs
        k_a = random_gen.random_sample(6) * 2 - 1
        # Calculate force and potential between each triplet in `triplets_idx`, using the already
        # tested function in `terms_single_array_simplified`
        *f012, pot012 = terms_single.angle_vibration_harmonic(
            q_i=q[0], q_j=q[1], q_k=q[2], angle0=angle0[0], k_a=k_a[0]
        )
        *f023, pot023 = terms_single.angle_vibration_harmonic(
            q_i=q[0], q_j=q[2], q_k=q[3], angle0=angle0[1], k_a=k_a[1]
        )
        *f034, pot034 = terms_single.angle_vibration_harmonic(
            q_i=q[0], q_j=q[3], q_k=q[4], angle0=angle0[2], k_a=k_a[2]
        )
        *f042, pot042 = terms_single.angle_vibration_harmonic(
            q_i=q[0], q_j=q[4], q_k=q[2], angle0=angle0[3], k_a=k_a[3]
        )
        *f123, pot123 = terms_single.angle_vibration_harmonic(
            q_i=q[1], q_j=q[2], q_k=q[3], angle0=angle0[4], k_a=k_a[4]
        )
        *f134, pot134 = terms_single.angle_vibration_harmonic(
            q_i=q[1], q_j=q[3], q_k=q[4], angle0=angle0[5], k_a=k_a[5]
        )
        # Calculate force and potential between each pair in `triplets_idx`, using the equivalent
        # function in `terms_multi_iter`
        f_calc, e_calc = terms_iter.angle_vibration_harmonic(
            q=q, triplets_idx=triplets_idx, angle0=angle0, k_a=k_a
        )
        # Verify that both functions' results are the same
        assert np.all(f_calc[0] == f012[0] + f023[0] + f034[0] + f042[0])
        assert np.all(f_calc[1] == f012[1] + f123[0] + f134[0])
        assert np.all(f_calc[2] == f012[2] + f023[1] + f042[2] + f123[1])
        assert np.all(f_calc[3] == f023[2] + f034[1] + f123[2] + f134[1])
        assert np.all(f_calc[4] == f034[2] + f042[1] + f134[2])
        assert e_calc[0] == pot012
        assert e_calc[1] == pot023
        assert e_calc[2] == pot034
        assert e_calc[3] == pot042
        assert e_calc[4] == pot123
        assert e_calc[5] == pot134
    return
