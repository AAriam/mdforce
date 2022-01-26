"""
Test functions for individual force-field terms in `mdforce.terms_single_simple`, evaluated using
analytical solutions obtained manually.
"""


# 3rd-party packages
import numpy as np

# Self
from mdforce import terms_single_simple as terms


def test_coulomb():
    """
    Test function for `mdforce.terms_single_simple.coulomb`.
    """
    # List of dicts, where each dict contains a set of chosen parameters for the coulomb function,
    # and the manually calculated values for potential 'e' and force components on particle 'i'.
    # To expand the test, add similar dicts to the list.
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
            "e_ij": 96.99484522,
            "f_i_x": -10.77720502,
            "f_i_y": -10.77720502,
            "f_i_z": -10.77720502,
        },
    ]

    # Iterate over sets of parameters/values
    for vals in val_lis:
        # Calculate potential between 'i' and 'j', and force on 'i', using the function
        f_i_x, f_i_y, f_i_z, e_ij = terms.coulomb(
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
        # Verify that the calculated results are the same as the manually obtained values
        assert np.isclose(e_ij, vals["e_ij"])
        assert np.isclose(f_i_x, vals["f_i_x"])
        assert np.isclose(f_i_y, vals["f_i_y"])
        assert np.isclose(f_i_z, vals["f_i_z"])

        # Calculate potential between 'j' and 'i', and force on 'j', using the function
        f_j_x, f_j_y, f_j_z, e_ji = terms.coulomb(
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
        # Verify that the potential remains the same, and the force on 'j' is equal to the force on
        # 'i' multiplied with -1
        assert np.isclose(e_ji, e_ij)
        assert np.isclose(f_j_x, -f_i_x)
        assert np.isclose(f_j_y, -f_i_y)
        assert np.isclose(f_j_z, -f_i_z)
    return


def test_lennard_jones():
    """
    Test function for `mdforce.terms_single_simple.lennard_jones`.
    """
    # List of dicts, where each dict contains a set of chosen parameters for the coulomb function,
    # and the manually calculated values for potential 'e' and force components on particle 'i'.
    # To expand the test, add similar dicts to the list.
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
            "e_ij": -4.064240392e-4,
            "f_i_x": 2.70937314e-4,
            "f_i_y": 2.70937314e-4,
            "f_i_z": 2.70937314e-4,
        },
    ]

    # Iterate over sets of parameters/values
    for vals in val_lis:
        # Calculate potential between 'i' and 'j', and force on 'i', using the function
        f_i_x, f_i_y, f_i_z, e_ij = terms.lennard_jones(
            q_i_x=vals["q_i_x"],
            q_i_y=vals["q_i_y"],
            q_i_z=vals["q_i_z"],
            q_j_x=vals["q_j_x"],
            q_j_y=vals["q_j_y"],
            q_j_z=vals["q_j_z"],
            a_ij=vals["a_ij"],
            b_ij=vals["b_ij"],
        )
        # Verify that the calculated results are the same as the manually obtained values
        assert np.isclose(e_ij, vals["e_ij"])
        assert np.isclose(f_i_x, vals["f_i_x"])
        assert np.isclose(f_i_y, vals["f_i_y"])
        assert np.isclose(f_i_z, vals["f_i_z"])

        # Calculate potential between 'j' and 'i', and force on 'j', using the function
        f_j_x, f_j_y, f_j_z, e_ji = terms.lennard_jones(
            q_i_x=vals["q_j_x"],
            q_i_y=vals["q_j_y"],
            q_i_z=vals["q_j_z"],
            q_j_x=vals["q_i_x"],
            q_j_y=vals["q_i_y"],
            q_j_z=vals["q_i_z"],
            a_ij=vals["a_ij"],
            b_ij=vals["b_ij"],
        )
        # Verify that the potential remains the same, and the force on 'j' is equal to the force on
        # 'i' multiplied with -1
        assert np.isclose(e_ji, e_ij)
        assert np.isclose(f_j_x, -f_i_x)
        assert np.isclose(f_j_y, -f_i_y)
        assert np.isclose(f_j_z, -f_i_z)
    return


def test_bond_vibration_harmonic():
    """
    Test function for `mdforce.terms_single_simple.bond_vibration_harmonic`.
    """
    # List of dicts, where each dict contains a set of chosen parameters for the coulomb function,
    # and the manually calculated values for potential 'e' and force components on particle 'i'.
    # To expand the test, add similar dicts to the list.
    val_lis = [
        {
            "q_i_x": 1,
            "q_i_y": 2,
            "q_i_z": 3,
            "q_j_x": 4,
            "q_j_y": 5,
            "q_j_z": 6,
            "d0": 8,
            "k_b": 7,
            "e_ij": 27.51546433,
            "f_i_x": -11.33161507,
            "f_i_y": -11.33161507,
            "f_i_z": -11.33161507,
        },
    ]

    # Iterate over sets of parameters/values
    for vals in val_lis:
        # Calculate potential between 'i' and 'j', and force on 'i', using the function
        f_i_x, f_i_y, f_i_z, e_ij = terms.bond_vibration_harmonic(
            q_i_x=vals["q_i_x"],
            q_i_y=vals["q_i_y"],
            q_i_z=vals["q_i_z"],
            q_j_x=vals["q_j_x"],
            q_j_y=vals["q_j_y"],
            q_j_z=vals["q_j_z"],
            d0=vals["d0"],
            k_b=vals["k_b"],
        )
        # Verify that the calculated results are the same as the manually obtained values
        assert np.isclose(e_ij, vals["e_ij"])
        assert np.isclose(f_i_x, vals["f_i_x"])
        assert np.isclose(f_i_y, vals["f_i_y"])
        assert np.isclose(f_i_z, vals["f_i_z"])

        # Calculate potential between 'j' and 'i', and force on 'j', using the function
        f_j_x, f_j_y, f_j_z, e_ji = terms.bond_vibration_harmonic(
            q_i_x=vals["q_j_x"],
            q_i_y=vals["q_j_y"],
            q_i_z=vals["q_j_z"],
            q_j_x=vals["q_i_x"],
            q_j_y=vals["q_i_y"],
            q_j_z=vals["q_i_z"],
            d0=vals["d0"],
            k_b=vals["k_b"],
        )
        # Verify that the potential remains the same, and the force on 'j' is equal to the force on
        # 'i' multiplied with -1
        assert np.isclose(e_ji, e_ij)
        assert np.isclose(f_j_x, -f_i_x)
        assert np.isclose(f_j_y, -f_i_y)
        assert np.isclose(f_j_z, -f_i_z)
    return


def test_angle_vibration_harmonic():
    """
    Test function for `mdforce.terms_single_simple.angle_vibration_harmonic`.
    """
    # List of dicts, where each dict contains a set of chosen parameters for the coulomb function,
    # and the manually calculated values for potential 'e' and force components on particle 'i'.
    # To expand the test, add similar dicts to the list.
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
            "angle0": 1,
            "e_ijk": 9.33523496574075,
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

    # Iterate over sets of parameters/values
    for vals in val_lis:
        # Calculate potential between 'i', 'j' and 'k', and force on each of them, using the
        # function
        (
            f_i_x,
            f_i_y,
            f_i_z,
            f_j_x,
            f_j_y,
            f_j_z,
            f_k_x,
            f_k_y,
            f_k_z,
            e_ijk,
        ) = terms.angle_vibration_harmonic(
            q_i_x=vals["q_i_x"],
            q_i_y=vals["q_i_y"],
            q_i_z=vals["q_i_z"],
            q_j_x=vals["q_j_x"],
            q_j_y=vals["q_j_y"],
            q_j_z=vals["q_j_z"],
            q_k_x=vals["q_k_x"],
            q_k_y=vals["q_k_y"],
            q_k_z=vals["q_k_z"],
            angle0=vals["angle0"],
            k_a=vals["k_a"],
        )
        # Verify that the calculated results are the same as the manually obtained values
        assert np.isclose(e_ijk, vals["e_ijk"])
        assert np.isclose(f_i_x, vals["f_i_x"])
        assert np.isclose(f_i_y, vals["f_i_y"])
        assert np.isclose(f_i_z, vals["f_i_z"])
        assert np.isclose(f_j_x, vals["f_j_x"])
        assert np.isclose(f_j_y, vals["f_j_y"])
        assert np.isclose(f_j_z, vals["f_j_z"])
        assert np.isclose(f_k_x, vals["f_k_x"])
        assert np.isclose(f_k_y, vals["f_k_y"])
        assert np.isclose(f_k_z, vals["f_k_z"])

        # Calculate potential between 'k', 'j' and 'i', and force on each of them, using the
        # function
        (
            f_k_x2,
            f_k_y2,
            f_k_z2,
            f_j_x2,
            f_j_y2,
            f_j_z2,
            f_i_x2,
            f_i_y2,
            f_i_z2,
            e_kji,
        ) = terms.angle_vibration_harmonic(
            q_i_x=vals["q_k_x"],
            q_i_y=vals["q_k_y"],
            q_i_z=vals["q_k_z"],
            q_j_x=vals["q_j_x"],
            q_j_y=vals["q_j_y"],
            q_j_z=vals["q_j_z"],
            q_k_x=vals["q_i_x"],
            q_k_y=vals["q_i_y"],
            q_k_z=vals["q_i_z"],
            angle0=vals["angle0"],
            k_a=vals["k_a"],
        )
        # Verify that swapping 'i' and 'k' gives the same potential and same force for 'j' (i.e.
        # the midlle particle), but swaps the forces of 'i' and 'k'
        assert np.isclose(e_kji, e_ijk)
        assert np.isclose(f_i_x2, f_i_x)
        assert np.isclose(f_i_y2, f_i_y)
        assert np.isclose(f_i_z2, f_i_z)
        assert np.isclose(f_j_x2, f_j_x)
        assert np.isclose(f_j_y2, f_j_y)
        assert np.isclose(f_j_z2, f_j_z)
        assert np.isclose(f_k_x2, f_k_x)
        assert np.isclose(f_k_y2, f_k_y)
        assert np.isclose(f_k_z2, f_k_z)
    return
