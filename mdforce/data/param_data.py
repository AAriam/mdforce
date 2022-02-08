"""
This module contains the correct dimensions and units of the force-field parameters as
`duq.Dimension` and `duq.Unit` objects, respectively.
"""


import duq


# Correct dimension of force-field parameters
dim_k_b = duq.Dimension("energy.length^-2")
dim_d0 = duq.Dimension("length")
dim_k_a = duq.Dimension("energy.dimensionless^-2")
dim_angle0 = duq.Dimension("dimensionless")
dim_lj_epsilon = duq.Dimension("energy")
dim_lj_sigma = duq.Dimension("length")
dim_lj_a = duq.Dimension("energy.length^12")
dim_lj_b = duq.Dimension("energy.length^6")
dim_c = duq.Dimension("electric charge")
dim_k_e = duq.Dimension("energy.length.electric charge^-2")
dim_m = duq.Dimension("mass")

# Internal units used in ForceField class
unit_mass = duq.Unit("Da")
unit_charge = duq.Unit("e")
unit_angle = duq.Unit("rad")
