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
            "pot": 96.99484522,
            "f_i_x": -10.77720502,
            "f_i_y": -10.77720502,
            "f_i_z": -10.77720502,
        },
    ]

    for vals in val_lis:
        # Calculate potential and force using the function
        f_i_x, f_i_y, f_i_z, pot_ij = terms.coulomb(
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
        assert np.isclose(pot_ij, vals["pot"])
        assert np.isclose(f_i_x, vals["f_i_x"])
        assert np.isclose(f_i_y, vals["f_i_y"])
        assert np.isclose(f_i_z, vals["f_i_z"])

        # Verify that swapping 'i' and 'j' gives the same potential, but negative force
        f_j_x, f_j_y, f_j_z, pot_ji = terms.coulomb(
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
        assert np.isclose(pot_ji, vals["pot"])
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
            "pot": -4.064240392e-4,
            "f_i_x": 2.70937314e-4,
            "f_i_y": 2.70937314e-4,
            "f_i_z": 2.70937314e-4,
        },
    ]

    for vals in val_lis:
        # Calculate potential and force using the function
        f_i_x, f_i_y, f_i_z, pot_ij = terms.lennard_jones(
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
        assert np.isclose(pot_ij, vals["pot"])
        assert np.isclose(f_i_x, vals["f_i_x"])
        assert np.isclose(f_i_y, vals["f_i_y"])
        assert np.isclose(f_i_z, vals["f_i_z"])

        # Verify that swapping 'i' and 'j' gives the same potential, but negative force
        f_j_x, f_j_y, f_j_z, pot_ji = terms.lennard_jones(
            q_i_x=vals["q_j_x"],
            q_i_y=vals["q_j_y"],
            q_i_z=vals["q_j_z"],
            q_j_x=vals["q_i_x"],
            q_j_y=vals["q_i_y"],
            q_j_z=vals["q_i_z"],
            a_ij=vals["a_ij"],
            b_ij=vals["b_ij"],
        )
        assert np.isclose(pot_ji, vals["pot"])
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
            "pot": 27.51546433,
            "f_i_x": -11.33161507,
            "f_i_y": -11.33161507,
            "f_i_z": -11.33161507,
        },
    ]

    for vals in val_lis:
        # Calculate potential and force using the function
        f_i_x, f_i_y, f_i_z, pot_ij = terms.bond_vibration_harmonic(
            q_i_x=vals["q_i_x"],
            q_i_y=vals["q_i_y"],
            q_i_z=vals["q_i_z"],
            q_j_x=vals["q_j_x"],
            q_j_y=vals["q_j_y"],
            q_j_z=vals["q_j_z"],
            k_b=vals["k_b"],
            dist_eq=vals["q_eq"],
        )

        # Compare the results
        assert np.isclose(pot_ij, vals["pot"])
        assert np.isclose(f_i_x, vals["f_i_x"])
        assert np.isclose(f_i_y, vals["f_i_y"])
        assert np.isclose(f_i_z, vals["f_i_z"])

        # Verify that swapping 'i' and 'j' gives the same potential, but negative force
        f_j_x, f_j_y, f_j_z, pot_ji = terms.bond_vibration_harmonic(
            q_i_x=vals["q_j_x"],
            q_i_y=vals["q_j_y"],
            q_i_z=vals["q_j_z"],
            q_j_x=vals["q_i_x"],
            q_j_y=vals["q_i_y"],
            q_j_z=vals["q_i_z"],
            k_b=vals["k_b"],
            dist_eq=vals["q_eq"],
        )
        assert np.isclose(pot_ji, vals["pot"])
        assert np.isclose(f_j_x, -vals["f_i_x"])
        assert np.isclose(f_j_y, -vals["f_i_y"])
        assert np.isclose(f_j_z, -vals["f_i_z"])
    return


def test_angle_vibration_harmonic():
    # List of dicts, where each dict contains a set of random parameters for the angle-vibration
    # function, and the manually calculated values for potential 'v' and force components on
    # all three particles. Particle `q_j` is the one in the middle.
    # To expand the test, just add similar dicts with other values to the list.
    val_lis = [
        {
            "q_i_x": 1,
            "q_i_y": 2,
            "q_i_z": 3,
            "q_j_x": 4,
            "q_j_y": 4,
            "q_j_z": 4,
            "q_k_x": 5,
            "q_k_y": 6,
            "q_k_z": 7,
            "k_a": 10,
            "angle_eq": 1,
            "pot": 9.33523496574075,
            "f_i_x": -1.5938004864718032,
            "f_i_y": 0.7969002432359013,
            "f_i_z": 3.1876009729436054,
            "f_j_x": 4.781401459415409,
            "f_j_y": 0,
            "f_j_z": -4.781401459415409,
            "f_k_x": -3.1876009729436054,
            "f_k_y": -0.7969002432359013,
            "f_k_z": 1.5938004864718032,
        },
    ]

    for vals in val_lis:
        # Calculate potential and force using the function
        (
            f_j_x,
            f_j_y,
            f_j_z,
            f_i_x,
            f_i_y,
            f_i_z,
            f_k_x,
            f_k_y,
            f_k_z,
            pot_ijk
        ) = terms.angle_vibration_harmonic(
            q_j_x=vals["q_j_x"],
            q_j_y=vals["q_j_y"],
            q_j_z=vals["q_j_z"],
            q_i_x=vals["q_i_x"],
            q_i_y=vals["q_i_y"],
            q_i_z=vals["q_i_z"],
            q_k_x=vals["q_k_x"],
            q_k_y=vals["q_k_y"],
            q_k_z=vals["q_k_z"],
            k_a=vals["k_a"],
            angle_eq=vals["angle_eq"],
        )

        # Compare the results
        assert np.isclose(pot_ijk, vals["pot"])
        assert np.isclose(f_i_x, vals["f_i_x"])
        assert np.isclose(f_i_y, vals["f_i_y"])
        assert np.isclose(f_i_z, vals["f_i_z"])
        assert np.isclose(f_j_x, vals["f_j_x"])
        assert np.isclose(f_j_y, vals["f_j_y"])
        assert np.isclose(f_j_z, vals["f_j_z"])
        assert np.isclose(f_k_x, vals["f_k_x"])
        assert np.isclose(f_k_y, vals["f_k_y"])
        assert np.isclose(f_k_z, vals["f_k_z"])

        # Verify that swapping 'i' and 'j' gives the same potential and same force for the middle
        # particle, but swaps the forces of 'i' and 'k'
        (
            f_j_x2,
            f_j_y2,
            f_j_z2,
            f_k_x2,
            f_k_y2,
            f_k_z2,
            f_i_x2,
            f_i_y2,
            f_i_z2,
            pot_kji
        ) = terms.angle_vibration_harmonic(
            q_j_x=vals["q_j_x"],
            q_j_y=vals["q_j_y"],
            q_j_z=vals["q_j_z"],
            q_i_x=vals["q_k_x"],
            q_i_y=vals["q_k_y"],
            q_i_z=vals["q_k_z"],
            q_k_x=vals["q_i_x"],
            q_k_y=vals["q_i_y"],
            q_k_z=vals["q_i_z"],
            k_a=vals["k_a"],
            angle_eq=vals["angle_eq"],
        )
        assert np.isclose(pot_ijk, pot_kji)
        assert np.isclose(f_i_x, f_i_x2)
        assert np.isclose(f_i_y, f_i_y2)
        assert np.isclose(f_i_z, f_i_z2)
        assert np.isclose(f_j_x, f_j_x2)
        assert np.isclose(f_j_y, f_j_y2)
        assert np.isclose(f_j_z, f_j_z2)
        assert np.isclose(f_k_x, f_k_x2)
        assert np.isclose(f_k_y, f_k_y2)
        assert np.isclose(f_k_z, f_k_z2)
    return
