"""
Test functions for the switch functions in `mdforce.switch_functions.py`.
"""

# 3rd-party packages
import numpy as np

# Self
from mdforce import switch_functions


# Set up random number generator with seed to make sure testing results are consistent
random_gen = np.random.RandomState(1111)


def test_poly1():
    for _ in range(10):
        d0, dc = random_gen.random_sample(size=2)
        poly1 = switch_functions.Poly1(d0, dc)
        for __ in range(10):
            q, d = random_gen.random_sample(size=2)
            s_ref, ds_ref = poly1._switch_test(q, d)
            s_opt, ds_opt = poly1(q, d)
            assert np.isclose(s_ref, s_opt)
            assert np.isclose(ds_ref, ds_opt)


def test_poly_1_behaviour():
    for _ in range(100):
        dc = random_gen.random_sample(size=1) * 10
        d0 = dc * random_gen.random_sample(size=1)
        switch = switch_functions.Poly1(d0, dc)
        # Create a random distance vector
        q = random_gen.random_sample(size=3)
        # Scale it to have a norm equal to d0
        q1 = q * d0 / np.linalg.norm(q)
        # Evaluate the switch function
        s1, ds1 = switch(q1, d0)
        # Verify that the switch is equal to 1 and its derivative is 0
        assert np.all(np.isclose(s1, 1))
        assert np.all(np.isclose(ds1, 0))
        # Verify that the value of switch is monotonously decreasing from d0 to dc
        for i in range(1, 100):
            d_i = d0 + (dc - d0) * i / 100
            q_i = q * d_i / np.linalg.norm(q)
            s_i, ds_i = switch(q_i, d_i)
            assert np.all(s_i < s1)
            s1 = s_i
        # Create a random distance vector
        q2 = random_gen.random_sample(size=3)
        # Scale it to have a norm equal to dc
        q2 = q2 * dc / np.linalg.norm(q2)
        # Evaluate the switch function
        s2, ds2 = switch(q2, dc)
        # Verify that the switch is equal to 0 and its derivative is 0
        assert np.all(np.isclose(s2, 0))
        assert np.all(np.isclose(ds2, 0))
    return
