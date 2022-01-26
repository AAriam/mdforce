"""
Test functions for individual force-field terms in `mdforce.terms_single_array_simplified`,
evaluated using functions in `mdforce.terms_single_array`.
"""


# 3rd-party packages
import numpy as np

# Self
from mdforce import terms_single_array as terms
from mdforce import terms_single_array_simplified as terms_simplified


# Set up random number generator with seed to make sure testing results are consistent
random_gen = np.random.RandomState(1111)


def test_coulomb():
    """
    Test function for `mdforce.terms_single_array_simplified.coulomb`.
    """
    # Run the test 100 times
    for i in range(100):
        # Create random x-y-z-coordinates for 2 particles 'i' and 'j'
        q_i = random_gen.random_sample(3) * 2 - 1
        q_j = random_gen.random_sample(3) * 2 - 1
        # Create random force constant, and charges for 'i' and 'j'
        k_e, c_i, c_j = random_gen.random_sample(3) * 2 - 1
        # Calculate potential and force on 'i' using the tested function in `terms_single_array`
        f_i, e_ij = terms.coulomb(q_i, q_j, c_i, c_j, k_e)
        # Calculate potential and force on 'i' using the equivalent function in
        # `terms_single_array_simplified` (to be tested)
        f_i_calc, e_ij_calc = terms_simplified.coulomb(q_i, q_j, c_i, c_j, k_e)
        # Verify that both functions return the same values for force and potential
        assert np.all(np.isclose(f_i_calc, f_i))
        assert np.isclose(e_ij_calc, e_ij)
    return


def test_lennard_jones():
    """
    Test function for `mdforce.terms_single_array_simplified.lennard_jones`.
    """
    # Run the test 100 times
    for i in range(100):
        # Create random x-y-z-coordinates for 2 particles 'i' and 'j'
        q_i = random_gen.random_sample(3) * 2 - 1
        q_j = random_gen.random_sample(3) * 2 - 1
        # Create random Lennard-Jones parameters A and B, for the pair of particles
        a, b = random_gen.random_sample(2) * 2 - 1
        # Calculate potential and force on 'i' using the tested function in `terms_single_array`
        f_i, e_ij = terms.lennard_jones(q_i, q_j, a, b)
        # Calculate potential and force on 'i' using the equivalent function in
        # `terms_single_array_simplified` (to be tested)
        f_i_calc, e_ij_calc = terms_simplified.lennard_jones(q_i, q_j, a, b)
        # Verify that both functions return the same values for force and potential
        assert np.all(np.isclose(f_i_calc, f_i))
        assert np.isclose(e_ij_calc, e_ij)
    return


def test_bond_vibration_harmonic():
    """
    Test function for `mdforce.terms_single_array_simplified.bond_vibration_harmonic`.
    """
    # Run the test 100 times
    for i in range(100):
        # Create random x-y-z-coordinates for 2 particles 'i' and 'j'
        q_i = random_gen.random_sample(3) * 2 - 1
        q_j = random_gen.random_sample(3) * 2 - 1
        # Create random equilibrium bond length and force constant for the pair of particles
        d0, k_b = random_gen.random_sample(2) * 2 - 1
        # Calculate potential and force on 'i' using the tested function in `terms_single_array`
        f_i, e_ij = terms.bond_vibration_harmonic(q_i, q_j, d0, k_b)
        # Calculate potential and force on 'i' using the equivalent function in
        # `terms_single_array_simplified` (to be tested)
        f_i_calc, e_ij_calc = terms_simplified.bond_vibration_harmonic(q_i, q_j, d0, k_b)
        # Verify that both functions return the same values for force and potential
        assert np.all(np.isclose(f_i_calc, f_i))
        assert np.isclose(e_ij_calc, e_ij)
    return


def test_angle_vibration_harmonic():
    """
    Test function for `mdforce.terms_single_array_simplified.angle_vibration_harmonic`.
    """
    # Run the test 100 times
    for i in range(100):
        # Create random x-y-z-coordinates for 2 particles 'i' and 'j'
        q_i = random_gen.random_sample(3) * 2 - 1
        q_j = random_gen.random_sample(3) * 2 - 1
        q_k = random_gen.random_sample(3) * 2 - 1
        # Create random equilibrium angle and force constant for the triplet of particles
        angle0, k_a = random_gen.random_sample(2) * 2 - 1
        # Calculate potential, and force on each particle using the tested function in
        # `terms_single_array`
        f_i, f_j, f_k, e_ijk = terms.angle_vibration_harmonic(q_i, q_j, q_k, angle0, k_a)
        # Calculate potential and force on each particle using the equivalent function in
        # `terms_single_array_simplified` (to be tested)
        f_i_calc, f_j_calc, f_k_calc, e_ijk_calc = terms_simplified.angle_vibration_harmonic(
            q_i, q_j, q_k, angle0, k_a
        )
        # Verify that both functions return the same values for force and potential
        assert np.all(np.isclose(f_i_calc, f_i))
        assert np.all(np.isclose(f_j_calc, f_j))
        assert np.all(np.isclose(f_k_calc, f_k))
        assert np.isclose(e_ijk_calc, e_ijk)
    return
