"""
Test functions for individual force-field terms in `mdforce.terms_single_array`, evaluated using
functions in `mdforce.terms_single_simple`.
"""

from mdforce import terms_single_simple as terms_simple
from mdforce import terms_single_array as terms_array
import numpy as np


def test_coulomb():
    """
    The formulas for coulomb potential 'V_e' as a function of two particles 'i' and
    'j', with position vectors 'q_i' and 'q_j', and charges 'c_i' and 'c_j' are:
    V_e(i, j) = k_e * c_i * c_j / ||r_ji||
    with the distance vector 'r_ji = q_i - q_j'.

    Consequently, the force vector 'F_e' on particle 'i' due to particle 'j' is:
    F_e(i, j) = k_e * c_i * c_j * (r_ji) / ||r_ji||^3
    """

    # Create sets of random parameters (in range [-1, 1)) 100 times
    for i in range(100):
        rand_vals = np.random.random(size=9) * 2 - 1
        q_i_x = rand_vals[0]
        q_i_y = rand_vals[1]
        q_i_z = rand_vals[2]
        q_j_x = rand_vals[3]
        q_j_y = rand_vals[4]
        q_j_z = rand_vals[5]
        c_i = rand_vals[6]
        c_j = rand_vals[7]
        k_e = rand_vals[8]

        # Calculate potential between particles 'i' and 'j', and the force on 'i' due to 'j',
        # using the already tested function in `terms_simple`
        v_ij, f_i_x, f_i_y, f_i_z = terms_simple.coulomb(
            q_i_x=q_i_x,
            q_i_y=q_i_y,
            q_i_z=q_i_z,
            q_j_x=q_j_x,
            q_j_y=q_j_y,
            q_j_z=q_j_z,
            c_i=c_i,
            c_j=c_j,
            k_e=k_e
        )

        # Create equivalent arrays for testing the equivalent function in `terms_array`
        q_i = np.array([q_i_x, q_i_y, q_i_z])
        q_j = np.array([q_j_x, q_j_y, q_j_z])

        # Call the function
        f_i_calc, v_ij_calc = terms_array.coulomb(q_i, q_j, c_i, c_j, k_e)
        # Test whether the function's results are the same
        assert np.isclose(v_ij_calc, v_ij)
        assert np.isclose(f_i_calc[0], f_i_x)
        assert np.isclose(f_i_calc[1], f_i_y)
        assert np.isclose(f_i_calc[2], f_i_z)

        # Verify that swapping 'i' and 'j' gives the same potential, but negative force
        f_j_calc, v_ji_calc = terms_array.coulomb(q_j, q_i, c_j, c_i, k_e)
        assert np.isclose(v_ji_calc, v_ij)
        assert np.isclose(f_j_calc[0], -f_i_x)
        assert np.isclose(f_j_calc[1], -f_i_y)
        assert np.isclose(f_j_calc[2], -f_i_z)
    return


def test_lennard_jones():
    pass


def test_bond_vibration_harmonic():
    pass


def test_angle_vibration_harmonic():
    pass