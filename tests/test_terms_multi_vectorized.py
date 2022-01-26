"""
Test functions for individual force-field terms in `mdforce.terms_multi_vectorized`, evaluated
using functions in `mdforce.terms_multi_iter`.
"""

# 3rd-party packages
import numpy as np

# Self
from mdforce import terms_multi_iter as terms_iter
from mdforce import terms_multi_vectorized as terms_vector


# Set up random number generator with seed to make sure testing results are consistent
random_gen = np.random.RandomState(1111)


def test_coulomb():
    # Create an index array of pairs, for which the interaction should be calculated
    pairs_idx = np.array([[0, 1], [0, 2], [0, 3], [0, 4]])
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
        f, pot = terms_iter.coulomb(q=q, pairs_idx=pairs_idx, c=c, k_e=k_e)

        # Calculate force and potential between q[0] and the rest, using the equivalent
        # function in `terms_multi_vectorized`
        f_i_total_calc, f_jsi_calc, pot_ijs_calc = terms_vector.coulomb(
            q[0], q[1:], c[0], c[1:], k_e
        )

        # Test whether both functions' results are the same
        assert np.all(np.isclose(f_i_total_calc, f[0]))
        assert np.all(np.isclose(f_jsi_calc[0], f[1]))
        assert np.all(np.isclose(f_jsi_calc[1], f[2]))
        assert np.all(np.isclose(f_jsi_calc[2], f[3]))
        assert np.all(np.isclose(f_jsi_calc[3], f[4]))
        assert np.all(np.isclose(pot_ijs_calc[0], pot[0]))
        assert np.all(np.isclose(pot_ijs_calc[1], pot[1]))
        assert np.all(np.isclose(pot_ijs_calc[2], pot[2]))
        assert np.all(np.isclose(pot_ijs_calc[3], pot[3]))
    return


def test_lennard_jones():
    # Create an index array of pairs, for which the interaction should be calculated
    pairs_idx = np.array([[0, 1], [0, 2], [0, 3], [0, 4]])
    # Create sets of random parameters (in range [-1, 1)) 100 times
    for i in range(100):
        # Create random x-y-z-coordinates for 5 particles
        q = random_gen.random_sample((5, 3)) * 2 - 1
        # Create random A-parameters for the 6 interacting pairs
        a = random_gen.random_sample(4) * 2 - 1
        # Create random B-parameters for the 6 interacting pairs
        b = random_gen.random_sample(4) * 2 - 1

        # Calculate force and potential between each pair in `pairs_idx`, using the already tested
        # function in `terms_single_array`
        f, pot = terms_iter.lennard_jones(q=q, pairs_idx=pairs_idx, a=a, b=b)

        # Calculate force and potential between q[0] and the rest, using the equivalent
        # function in `terms_multi_vectorized`
        f_i_total_calc, f_jsi_calc, pot_ijs_calc = terms_vector.lennard_jones(q[0], q[1:], a, b)

        # Test whether both functions' results are the same
        assert np.all(np.isclose(f_i_total_calc, f[0]))
        assert np.all(np.isclose(f_jsi_calc[0], f[1]))
        assert np.all(np.isclose(f_jsi_calc[1], f[2]))
        assert np.all(np.isclose(f_jsi_calc[2], f[3]))
        assert np.all(np.isclose(f_jsi_calc[3], f[4]))
        assert np.all(np.isclose(pot_ijs_calc[0], pot[0]))
        assert np.all(np.isclose(pot_ijs_calc[1], pot[1]))
        assert np.all(np.isclose(pot_ijs_calc[2], pot[2]))
        assert np.all(np.isclose(pot_ijs_calc[3], pot[3]))
    return


def test_bond_vibration_harmonic():
    # Create an index array of pairs, for which the interaction should be calculated
    pairs_idx = np.array([[0, 1], [0, 2], [0, 3], [0, 4]])
    # Create sets of random parameters (in range [-1, 1)) 100 times
    for i in range(100):
        # Create random x-y-z-coordinates for 5 particles
        q = random_gen.random_sample((5, 3)) * 2 - 1
        # Create random equilibrium bond distances for the 6 interacting pairs
        dist_eq = random_gen.random_sample(4) * 2 - 1
        # Create random force constants for the 6 interacting pairs
        k_b = random_gen.random_sample(4) * 2 - 1

        # Calculate force and potential between each pair in `pairs_idx`, using the already tested
        # function in `terms_single_array`
        f, pot = terms_iter.bond_vibration_harmonic(
            q=q, pairs_idx=pairs_idx, dist_eq=dist_eq, k_b=k_b
        )

        # Calculate force and potential between q[0] and the rest, using the equivalent
        # function in `terms_multi_vectorized`
        f_i_total_calc, f_jsi_calc, pot_ijs_calc = terms_vector.bond_vibration_harmonic(
            q[0], q[1:], dist_eq, k_b
        )

        # Test whether both functions' results are the same
        assert np.all(np.isclose(f_i_total_calc, f[0]))
        assert np.all(np.isclose(f_jsi_calc[0], f[1]))
        assert np.all(np.isclose(f_jsi_calc[1], f[2]))
        assert np.all(np.isclose(f_jsi_calc[2], f[3]))
        assert np.all(np.isclose(f_jsi_calc[3], f[4]))
        assert np.all(np.isclose(pot_ijs_calc[0], pot[0]))
        assert np.all(np.isclose(pot_ijs_calc[1], pot[1]))
        assert np.all(np.isclose(pot_ijs_calc[2], pot[2]))
        assert np.all(np.isclose(pot_ijs_calc[3], pot[3]))
    return
