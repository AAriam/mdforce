import numpy as np

from ._find_interacting_partners import find_interacting_partners_indices
from .... import terms_iter as terms


def parametrize(param, mol_ids, atom_ids, bonded_atoms_idx):
    (pair_idx_lennard_jones,
     pair_idx_coulomb,
     pair_idx_bond_vib,
     pair_idx_angle_vib
     ) = find_interacting_partners_indices(
        mol_ids,
        atom_ids,
        bonded_atoms_idx
    )

    params = param.value["data"]

    lj_ab = np.zeros_like(pair_idx_lennard_jones)
    lj_ab[:,0] = params["LJ_A"]
    lj_ab[:,1] = params["LJ_B"]

    coulomb_pairs_atom_ids = atom_ids[pair_idx_coulomb]
    coulomb_c = np.where(coulomb_pairs_atom_ids==1, params["CHARGE_H"], params["CHARGE_O"])

    bond_eq_dist = np.full(pair_idx_bond_vib.shape[0], params["BOND_EQ_DIST"])

    angle_eq_angle = np.full(pair_idx_angle_vib.shape[0], params["ANGLE_EQ_ANGLE"])

    def force_field(q):
        f1, e1 = terms.lennard_jones(q, pair_idx_lennard_jones, lj_ab)
        f2, e2 = terms.coulomb(q, pair_idx_coulomb, coulomb_c, params["COULOMB_K"])
        f3, e3 = terms.bond_vibration_harmonic(q, pair_idx_bond_vib, bond_eq_dist, params["BOND_K"])
        # FIXME
        # remove angle
        f4, e4, angle = terms.angle_vibration_harmonic(q, pair_idx_angle_vib, angle_eq_angle, params["ANGLE_EQ_ANGLE"])
        return f1, f2, f3, f4, e1, e2, e3, e4, angle

    return force_field
