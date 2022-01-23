
import duq


class ForceField:

    # Correct dimension of force-field parameters
    _dim_bond_vib_k = duq.Dimension("energy.length^-2")
    _dim_bond_eq_dist = duq.Dimension("length")
    _dim_angle_vib_k = duq.Dimension("energy.dimensionless^-2")
    _dim_angle_eq_angle = duq.Dimension("dimensionless")
    _dim_lj_epsilon = duq.Dimension("energy")
    _dim_lj_sigma = duq.Dimension("length")
    _dim_charge = duq.Dimension("electric charge")
    _dim_coulomb_k = duq.Dimension("energy.length.electric charge^-2")
    _dim_mass = duq.Dimension("mass")
