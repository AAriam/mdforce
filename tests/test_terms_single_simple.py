"""
Test functions for individual force-field terms in `mdforce.terms_single_simple`, evaluated using
analytical solutions obtained manually.
"""

from mdforce import terms_single_simple as terms
import numpy as np


def test_coulomb():

    # List of dicts, where each dict contains a set of random parameters for the coulomb function,
    # and the manually calculated values for potential 'v' and force components on particle 'i'.
    # To expand the test, just add similar dicts with other values to the list.
    val_lis = [
        {
            "q_i_x": 1,
            "q_i_y": 2,
            "q_i_z": 3,
            "q_j_x": 4,
            "q_j_y": 5,
            "q_j_z": 6,
            "c_i": 7,
            "c_j": 8,
            "k_e": 9,
            "v": 96.99484522,
            "f_i_x": -10.77720502,
            "f_i_y": -10.77720502,
            "f_i_z": -10.77720502,
        },
    ]

    for vals in val_lis:
        # Calculate potential and force using the function
        v_ij, f_i_x, f_i_y, f_i_z = terms.coulomb(
            q_i_x=vals["q_i_x"],
            q_i_y=vals["q_i_y"],
            q_i_z=vals["q_i_z"],
            q_j_x=vals["q_j_x"],
            q_j_y=vals["q_j_y"],
            q_j_z=vals["q_j_z"],
            c_i=vals["c_i"],
            c_j=vals["c_j"],
            k_e=vals["k_e"],
        )

        # Compare the results
        assert np.isclose(v_ij, vals["v"])
        assert np.isclose(f_i_x, vals["f_i_x"])
        assert np.isclose(f_i_y, vals["f_i_y"])
        assert np.isclose(f_i_z, vals["f_i_z"])

        # Verify that swapping 'i' and 'j' gives the same potential, but negative force
        v_ji, f_j_x, f_j_y, f_j_z = terms.coulomb(
            q_i_x=vals["q_j_x"],
            q_i_y=vals["q_j_y"],
            q_i_z=vals["q_j_z"],
            q_j_x=vals["q_i_x"],
            q_j_y=vals["q_i_y"],
            q_j_z=vals["q_i_z"],
            c_i=vals["c_j"],
            c_j=vals["c_i"],
            k_e=vals["k_e"],
        )
        assert np.isclose(v_ji, vals["v"])
        assert np.isclose(f_j_x, -vals["f_i_x"])
        assert np.isclose(f_j_y, -vals["f_i_y"])
        assert np.isclose(f_j_z, -vals["f_i_z"])
    return


def test_lennard_jones():

    # List of dicts, where each dict contains a set of random parameters for the LJ function,
    # and the manually calculated values for potential 'v' and force components on particle 'i'.
    # To expand the test, just add similar dicts with other values to the list.
    val_lis = [
        {
            "q_i_x": 1,
            "q_i_y": 2,
            "q_i_z": 3,
            "q_j_x": 4,
            "q_j_y": 5,
            "q_j_z": 6,
            "a_ij": 7,
            "b_ij": 8,
            "v": -4.064240392e-4,
            "f_i_x": 2.70937314e-4,
            "f_i_y": 2.70937314e-4,
            "f_i_z": 2.70937314e-4,
        },
    ]

    for vals in val_lis:
        # Calculate potential and force using the function
        v_ij, f_i_x, f_i_y, f_i_z = terms.lennard_jones(
            q_i_x=vals["q_i_x"],
            q_i_y=vals["q_i_y"],
            q_i_z=vals["q_i_z"],
            q_j_x=vals["q_j_x"],
            q_j_y=vals["q_j_y"],
            q_j_z=vals["q_j_z"],
            a_ij=vals["a_ij"],
            b_ij=vals["b_ij"],
        )

        # Compare the results
        assert np.isclose(v_ij, vals["v"])
        assert np.isclose(f_i_x, vals["f_i_x"])
        assert np.isclose(f_i_y, vals["f_i_y"])
        assert np.isclose(f_i_z, vals["f_i_z"])

        # Verify that swapping 'i' and 'j' gives the same potential, but negative force
        v_ji, f_j_x, f_j_y, f_j_z = terms.lennard_jones(
            q_i_x=vals["q_j_x"],
            q_i_y=vals["q_j_y"],
            q_i_z=vals["q_j_z"],
            q_j_x=vals["q_i_x"],
            q_j_y=vals["q_i_y"],
            q_j_z=vals["q_i_z"],
            a_ij=vals["a_ij"],
            b_ij=vals["b_ij"],
        )
        assert np.isclose(v_ji, vals["v"])
        assert np.isclose(f_j_x, -vals["f_i_x"])
        assert np.isclose(f_j_y, -vals["f_i_y"])
        assert np.isclose(f_j_z, -vals["f_i_z"])
    return


def test_bond_vibration_harmonic():
    # List of dicts, where each dict contains a set of random parameters for the bond-vibration
    # function, and the manually calculated values for potential 'v' and force components on
    # particle 'i'.
    # To expand the test, just add similar dicts with other values to the list.
    val_lis = [
        {
            "q_i_x": 1,
            "q_i_y": 2,
            "q_i_z": 3,
            "q_j_x": 4,
            "q_j_y": 5,
            "q_j_z": 6,
            "k_b": 7,
            "q_eq": 8,
            "v": 27.51546433,
            "f_i_x": -11.33161507,
            "f_i_y": -11.33161507,
            "f_i_z": -11.33161507,
        },
    ]

    for vals in val_lis:
        # Calculate potential and force using the function
        v_ij, f_i_x, f_i_y, f_i_z = terms.bond_vibration_harmonic(
            q_i_x=vals["q_i_x"],
            q_i_y=vals["q_i_y"],
            q_i_z=vals["q_i_z"],
            q_j_x=vals["q_j_x"],
            q_j_y=vals["q_j_y"],
            q_j_z=vals["q_j_z"],
            k_b=vals["k_b"],
            q_eq=vals["q_eq"],
        )

        # Compare the results
        assert np.isclose(v_ij, vals["v"])
        assert np.isclose(f_i_x, vals["f_i_x"])
        assert np.isclose(f_i_y, vals["f_i_y"])
        assert np.isclose(f_i_z, vals["f_i_z"])

        # Verify that swapping 'i' and 'j' gives the same potential, but negative force
        v_ji, f_j_x, f_j_y, f_j_z = terms.bond_vibration_harmonic(
            q_i_x=vals["q_j_x"],
            q_i_y=vals["q_j_y"],
            q_i_z=vals["q_j_z"],
            q_j_x=vals["q_i_x"],
            q_j_y=vals["q_i_y"],
            q_j_z=vals["q_i_z"],
            k_b=vals["k_b"],
            q_eq=vals["q_eq"],
        )
        assert np.isclose(v_ji, vals["v"])
        assert np.isclose(f_j_x, -vals["f_i_x"])
        assert np.isclose(f_j_y, -vals["f_i_y"])
        assert np.isclose(f_j_z, -vals["f_i_z"])
    return
