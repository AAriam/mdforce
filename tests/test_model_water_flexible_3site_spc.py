"""
Test functions for the water model in `mdforce.models.water.flexible_3site_spc`, evaluated
using functions in `mdforce.terms_multi_vectorized`.
"""


# 3rd-party packages
import numpy as np

# Self
from mdforce import terms_multi_vectorized as terms_vector
from mdforce.models.water.flexible_3site_spc import Flexible3SiteSPC


# Set up random number generator with seed to make sure testing results are consistent
random_gen = np.random.RandomState(1111)


def test_model_water_flexible_3site_spc():
    """
    Test function for `mdforce.models.water.flexible_3site_spc.Flexible3SiteSPC`.
    """
    # Run the test 100 times with random arguments
    for _ in range(10):
        # Create random force-field parameters
        (
            k_b,
            d0,
            k_a,
            angle0,
            lj_epsilon,
            lj_sigma,
            c_o,
            c_h,
            k_e,
            m_o,
            m_h,
        ) = random_gen.random_sample(size=11)
        # Instantiate the force-field with those parameters
        forcefield = Flexible3SiteSPC(
            k_b, d0, k_a, angle0, lj_epsilon, lj_sigma, c_o, c_h, k_e, m_o, m_h
        )
        # Define the shape of data (in this case, 9 atoms in 3D-space)
        shape_data = (9, 3)
        # Prepare the force-field for this shape of data
        forcefield.initialize_forcefield(shape_data)

        c = np.tile([c_o, c_h, c_h], 3)

        lj_a = 4 * lj_epsilon * lj_sigma ** 12
        lj_b = 4 * lj_epsilon * lj_sigma ** 6

        lj_as = np.repeat(lj_a, 2)
        lj_bs = np.repeat(lj_b, 2)

        d0s = np.array([d0])
        k_bs = np.array([k_b])

        for _ in range(100):
            # Create an array of random coordinates
            q = random_gen.random_sample(size=shape_data)
            # Evaluate all forces
            forcefield(q)

            # ---------- Test coulomb ----------
            f_coulomb = np.zeros(shape_data)
            f_c_0, f_c_38_0, e_c_38_0 = terms_vector.coulomb(q[0], q[3:], c[0], c[3:], k_e)
            f_c_1, f_c_38_1, e_c_38_1 = terms_vector.coulomb(q[1], q[3:], c[1], c[3:], k_e)
            f_c_2, f_c_38_2, e_c_38_2 = terms_vector.coulomb(q[2], q[3:], c[2], c[3:], k_e)
            f_c_3, f_c_68_3, e_c_68_3 = terms_vector.coulomb(q[3], q[6:], c[3], c[6:], k_e)
            f_c_4, f_c_68_4, e_c_68_4 = terms_vector.coulomb(q[4], q[6:], c[4], c[6:], k_e)
            f_c_5, f_c_68_5, e_c_68_5 = terms_vector.coulomb(q[5], q[6:], c[5], c[6:], k_e)
            e_coulomb = (
                e_c_38_0.sum()
                + e_c_38_1.sum()
                + e_c_38_2.sum()
                + e_c_68_3.sum()
                + e_c_68_4.sum()
                + e_c_68_5.sum()
            )
            f_coulomb[0] = f_c_0
            f_coulomb[1] = f_c_1
            f_coulomb[2] = f_c_2
            f_coulomb[3] = f_c_3 + f_c_38_0[0] + f_c_38_1[0] + f_c_38_2[0]
            f_coulomb[4] = f_c_4 + f_c_38_0[1] + f_c_38_1[1] + f_c_38_2[1]
            f_coulomb[5] = f_c_5 + f_c_38_0[2] + f_c_38_1[2] + f_c_38_2[2]
            f_coulomb[6] = (
                f_c_38_0[3] + f_c_38_1[3] + f_c_38_2[3] + f_c_68_3[0] + f_c_68_4[0] + f_c_68_5[0]
            )
            f_coulomb[7] = (
                f_c_38_0[4] + f_c_38_1[4] + f_c_38_2[4] + f_c_68_3[1] + f_c_68_4[1] + f_c_68_5[1]
            )
            f_coulomb[8] = (
                f_c_38_0[5] + f_c_38_1[5] + f_c_38_2[5] + f_c_68_3[2] + f_c_68_4[2] + f_c_68_5[2]
            )
            assert np.all(np.isclose(forcefield.force_coulomb, f_coulomb))
            assert np.isclose(forcefield.energy_coulomb, e_coulomb)

            # ---------- Test Lennard-Jones ----------
            f_lj = np.zeros(shape_data)
            f_lj_0, f_lj_36, e_lj_36_0 = terms_vector.lennard_jones(
                q[0], q[[3, 6]], lj_as[[0, 1]], lj_bs[[0, 1]]
            )
            f_lj_3, f_lj_6, e_lj_6_3 = terms_vector.lennard_jones(
                q[3], q[[6]], lj_as[[1]], lj_bs[[1]]
            )
            f_lj[0] = f_lj_0
            f_lj[3] = f_lj_3 + f_lj_36[0]
            f_lj[6] = f_lj_36[1] + f_lj_6
            e_lj = e_lj_36_0.sum() + e_lj_6_3.sum()
            assert np.all(np.isclose(forcefield.force_lennard_jones, f_lj))
            assert np.isclose(forcefield.energy_lennard_jones, e_lj)

            # ---------- Test bond-vibration ----------
            f_bond = np.zeros(shape_data)
            f_b_0, f_b_12, e_b_12_0 = terms_vector.bond_vibration_harmonic(
                q[0], q[[1, 2]], d0s, k_bs
            )
            f_b_3, f_b_45, e_b_45_3 = terms_vector.bond_vibration_harmonic(
                q[3], q[[4, 5]], d0s, k_bs
            )
            f_b_6, f_b_78, e_b_78_6 = terms_vector.bond_vibration_harmonic(
                q[6], q[[7, 8]], d0s, k_bs
            )
            f_bond[0] = f_b_0
            f_bond[[1, 2]] = f_b_12
            f_bond[3] = f_b_3
            f_bond[[4, 5]] = f_b_45
            f_bond[6] = f_b_6
            f_bond[[7, 8]] = f_b_78
            e_bond = e_b_12_0.sum() + e_b_45_3.sum() + e_b_78_6.sum()
            assert np.all(np.isclose(forcefield.force_bond_vibration, f_bond))
            assert np.isclose(forcefield.energy_bond_vibration, e_bond)

            # Test angle-vibration
            f_angle = np.zeros(shape_data)
            from mdforce.terms_single_array_simplified import angle_vibration_harmonic

            f_angle[1], f_angle[0], f_angle[2], e_a_12_0 = angle_vibration_harmonic(
                q[1], q[0], q[2], angle0, k_a
            )
            f_angle[4], f_angle[3], f_angle[5], e_a_45_3 = angle_vibration_harmonic(
                q[4], q[3], q[5], angle0, k_a
            )
            f_angle[7], f_angle[6], f_angle[8], e_a_67_8 = angle_vibration_harmonic(
                q[7], q[6], q[8], angle0, k_a
            )
            e_angle = e_a_12_0 + e_a_45_3 + e_a_67_8
            assert np.all(np.isclose(forcefield.force_angle_vibration, f_angle))
            assert np.isclose(forcefield.energy_angle_vibration, e_angle)

            # Verify that the total force is the same
            f_total = f_coulomb + f_lj + f_bond + f_angle
            assert np.all(np.isclose(forcefield.force_total, f_total))

            # Verify that the acceleration is the same
            masses = np.tile([m_o, m_h, m_h], 3).reshape(-1, 1)
            assert np.all(np.isclose(forcefield.acceleration, f_total / masses))
