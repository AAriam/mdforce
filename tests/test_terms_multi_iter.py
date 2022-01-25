"""
Test functions for individual force-field terms in `mdforce.terms_multi_iter`, evaluated using
functions in `mdforce.terms_single_array`.
"""

# 3rd-party packages
import numpy as np
# Self
from mdforce import terms_single_array as terms_single
from mdforce import terms_multi_iter as terms_iter


# Set up random number generator with seed to make sure testing results are consistent
random_gen = np.random.RandomState(1111)


def test_coulomb():
    # Create an index array of pairs, for which the interaction should be calculated
    pairs_idx = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3]])
    # Create sets of random parameters (in range [-1, 1)) 100 times
    for i in range(100):
        # Create random x-y-z-coordinates for 5 particles
        q = random_gen.random_sample((5, 3)) * 2 - 1
        # Create random charges for 5 particles
        c = random_gen.random_sample(5) * 2 - 1
        # Create random force constant
        k_e = random_gen.random_sample() * 2 - 1

        # Calculate force and potential between each pair in `pairs_idx`, using the already tested
        # function in `terms_single_array`
        f01, pot01 = terms_single.coulomb(q[0], q[1], c[0], c[1], k_e)
        f02, pot02 = terms_single.coulomb(q[0], q[2], c[0], c[2], k_e)
        f03, pot03 = terms_single.coulomb(q[0], q[3], c[0], c[3], k_e)
        f04, pot04 = terms_single.coulomb(q[0], q[4], c[0], c[4], k_e)
        f12, pot12 = terms_single.coulomb(q[1], q[2], c[1], c[2], k_e)
        f13, pot13 = terms_single.coulomb(q[1], q[3], c[1], c[3], k_e)

        # Calculate force and potential between each pair in `pairs_idx`, using the equivalent
        # function in `terms_multi_iter`
        f_calc, pot_calc = terms_iter.coulomb(q=q, pairs_idx=pairs_idx, c=c, k_e=k_e)

        # Test whether both functions' results are the same
        assert np.all(f_calc[0] == f01 + f02 + f03 + f04)
        assert np.all(f_calc[1] == -f01 + f12 + f13)
        assert np.all(f_calc[2] == -f02 - f12)
        assert np.all(f_calc[3] == -f03 - f13)
        assert np.all(f_calc[4] == -f04)
        assert pot_calc[0] == pot01
        assert pot_calc[1] == pot02
        assert pot_calc[2] == pot03
        assert pot_calc[3] == pot04
        assert pot_calc[4] == pot12
        assert pot_calc[5] == pot13
        return


def test_lennard_jones():
    # Create an index array of pairs, for which the interaction should be calculated
    pairs_idx = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3]])
    # Create sets of random parameters (in range [-1, 1)) 100 times
    for i in range(100):
        # Create random x-y-z-coordinates for 5 particles
        q = random_gen.random_sample((5, 3)) * 2 - 1
        # Create random A-parameters for the 6 interacting pairs
        a = random_gen.random_sample(6) * 2 - 1
        # Create random B-parameters for the 6 interacting pairs
        b = random_gen.random_sample(6) * 2 - 1

        # Calculate force and potential between each pair in `pairs_idx`, using the already tested
        # function in `terms_single_array`
        f01, pot01 = terms_single.lennard_jones(q_i=q[0], q_j=q[1], a_ij=a[0], b_ij=b[0])
        f02, pot02 = terms_single.lennard_jones(q_i=q[0], q_j=q[2], a_ij=a[1], b_ij=b[1])
        f03, pot03 = terms_single.lennard_jones(q_i=q[0], q_j=q[3], a_ij=a[2], b_ij=b[2])
        f04, pot04 = terms_single.lennard_jones(q_i=q[0], q_j=q[4], a_ij=a[3], b_ij=b[3])
        f12, pot12 = terms_single.lennard_jones(q_i=q[1], q_j=q[2], a_ij=a[4], b_ij=b[4])
        f13, pot13 = terms_single.lennard_jones(q_i=q[1], q_j=q[3], a_ij=a[5], b_ij=b[5])

        # Calculate force and potential between each pair in `pairs_idx`, using the equivalent
        # function in `terms_multi_iter`
        f_calc, pot_calc = terms_iter.lennard_jones(q=q, pairs_idx=pairs_idx, a=a, b=b)

        # Test whether both functions' results are the same
        assert np.all(f_calc[0] == f01 + f02 + f03 + f04)
        assert np.all(f_calc[1] == -f01 + f12 + f13)
        assert np.all(f_calc[2] == -f02 - f12)
        assert np.all(f_calc[3] == -f03 - f13)
        assert np.all(f_calc[4] == -f04)
        assert pot_calc[0] == pot01
        assert pot_calc[1] == pot02
        assert pot_calc[2] == pot03
        assert pot_calc[3] == pot04
        assert pot_calc[4] == pot12
        assert pot_calc[5] == pot13
        return


def test_bond_vibration_harmonic():
    # Create an index array of pairs, for which the interaction should be calculated
    pairs_idx = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3]])
    # Create sets of random parameters (in range [-1, 1)) 100 times
    for i in range(100):
        # Create random x-y-z-coordinates for 5 particles
        q = random_gen.random_sample((5, 3)) * 2 - 1
        # Create random equilibrium bond distances for the 6 interacting pairs
        dist_eq = random_gen.random_sample(6) * 2 - 1
        # Create random force constants for the 6 interacting pairs
        k_b = random_gen.random_sample(6) * 2 - 1

        # Calculate force and potential between each pair in `pairs_idx`, using the already tested
        # function in `terms_single_array`
        f01, pot01 = terms_single.bond_vibration_harmonic(q_i=q[0], q_j=q[1], dist_eq=dist_eq[0], k_b=k_b[0])
        f02, pot02 = terms_single.bond_vibration_harmonic(q_i=q[0], q_j=q[2], dist_eq=dist_eq[1], k_b=k_b[1])
        f03, pot03 = terms_single.bond_vibration_harmonic(q_i=q[0], q_j=q[3], dist_eq=dist_eq[2], k_b=k_b[2])
        f04, pot04 = terms_single.bond_vibration_harmonic(q_i=q[0], q_j=q[4], dist_eq=dist_eq[3], k_b=k_b[3])
        f12, pot12 = terms_single.bond_vibration_harmonic(q_i=q[1], q_j=q[2], dist_eq=dist_eq[4], k_b=k_b[4])
        f13, pot13 = terms_single.bond_vibration_harmonic(q_i=q[1], q_j=q[3], dist_eq=dist_eq[5], k_b=k_b[5])

        # Calculate force and potential between each pair in `pairs_idx`, using the equivalent
        # function in `terms_multi_iter`
        f_calc, pot_calc = terms_iter.bond_vibration_harmonic(
            q=q, pairs_idx=pairs_idx, dist_eq=dist_eq, k_b=k_b
        )

        # Test whether both functions' results are the same
        assert np.all(f_calc[0] == f01 + f02 + f03 + f04)
        assert np.all(f_calc[1] == -f01 + f12 + f13)
        assert np.all(f_calc[2] == -f02 - f12)
        assert np.all(f_calc[3] == -f03 - f13)
        assert np.all(f_calc[4] == -f04)
        assert pot_calc[0] == pot01
        assert pot_calc[1] == pot02
        assert pot_calc[2] == pot03
        assert pot_calc[3] == pot04
        assert pot_calc[4] == pot12
        assert pot_calc[5] == pot13
        return


def test_angle_vibration_harmonic():
    # Create an index array of triplets, for which the interaction should be calculated
    triplets_idx = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 2], [1, 2, 3], [1, 3, 4]])
    # Create sets of random parameters (in range [-1, 1)) 100 times
    for i in range(100):
        # Create random x-y-z-coordinates for 5 particles
        q = random_gen.random_sample((5, 3)) * 2 - 1
        # Create random equilibrium bond distances for the 6 interacting pairs
        angle_eq = random_gen.random_sample(6) * 2 - 1
        # Create random force constants for the 6 interacting pairs
        k_a = random_gen.random_sample(6) * 2 - 1

        # Calculate force and potential between each triplet in `triplets_idx`, using the already
        # tested function in `terms_single_array`
        *f012, pot012 = terms_single.angle_vibration_harmonic(q_j=q[0], q_i=q[1], q_k=q[2], angle_eq=angle_eq[0], k_a=k_a[0])
        *f023, pot023 = terms_single.angle_vibration_harmonic(q_j=q[0], q_i=q[2], q_k=q[3], angle_eq=angle_eq[1], k_a=k_a[1])
        *f034, pot034 = terms_single.angle_vibration_harmonic(q_j=q[0], q_i=q[3], q_k=q[4], angle_eq=angle_eq[2], k_a=k_a[2])
        *f042, pot042 = terms_single.angle_vibration_harmonic(q_j=q[0], q_i=q[4], q_k=q[2], angle_eq=angle_eq[3], k_a=k_a[3])
        *f123, pot123 = terms_single.angle_vibration_harmonic(q_j=q[1], q_i=q[2], q_k=q[3], angle_eq=angle_eq[4], k_a=k_a[4])
        *f134, pot134 = terms_single.angle_vibration_harmonic(q_j=q[1], q_i=q[3], q_k=q[4], angle_eq=angle_eq[5], k_a=k_a[5])

        # Calculate force and potential between each pair in `triplets_idx`, using the equivalent
        # function in `terms_multi_iter`
        f_calc, pot_calc = terms_iter.angle_vibration_harmonic(
            q=q, triplets_idx=triplets_idx, angle_eq=angle_eq, k_a=k_a
        )

        # Test whether both functions' results are the same
        assert np.all(f_calc[0] == f012[0] + f023[0] + f034[0] + f042[0])
        assert np.all(f_calc[1] == f012[1] + f123[0] + f134[0])
        assert np.all(f_calc[2] == f012[2] + f023[1] + f042[2] + f123[1])
        assert np.all(f_calc[3] == f023[2] + f034[1] + f123[2] + f134[1])
        assert np.all(f_calc[4] == f034[2] + f042[1] + f134[2])
        assert pot_calc[0] == pot012
        assert pot_calc[1] == pot023
        assert pot_calc[2] == pot034
        assert pot_calc[3] == pot042
        assert pot_calc[4] == pot123
        assert pot_calc[5] == pot134
        return