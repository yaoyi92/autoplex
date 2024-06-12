from __future__ import annotations
import pytest
from ase.io import read
from ase.atoms import Atom
from autoplex.fitting.common.regularization import (
    set_sigma
)


def test_set_sigma(test_dir):
    # data setup
    test_atoms = read(test_dir / 'fitting/pre_xyz_train_more_data.extxyz', ':')
    isol_es = {3: -0.28649227, 17: -0.28649227}
    reg_minmax = [(0.1, 1), (0.001, 0.1), 
                  (0.0316, 0.316), 
                  (0.0632, 0.632)]
    
    # test series of options for set_sigma
    
    atoms_modi = set_sigma(test_atoms, 
              reg_minmax,
              scheme='linear-hull',)
    assert atoms_modi[2].info['energy_sigma'] == 0.001
    
    
    atoms_modi = set_sigma(test_atoms, 
              reg_minmax,
              scheme='linear-hull',
              config_type_override={'test': [1e-4, 1e-4, 1e-4]}
              )
    assert atoms_modi[2].info['energy_sigma'] == 1e-4
    
    
    atoms_modi[0].info['REF_energy'] += 20
    for atoms in atoms_modi[:3]:
        atoms.set_cell([10, 10, 10])
    for atoms in atoms_modi[4:]:
        atoms.set_cell([11, 11, 11])
    atoms_modi = set_sigma(test_atoms, 
              reg_minmax,
              scheme='linear-hull',
              max_energy=0.05,
              isol_es=isol_es
              )
    assert len(atoms_modi) < len(test_atoms)
    
    
    atoms_modi[0].append(Atom('Li', [1,1,1]))
    atoms_modi = set_sigma(test_atoms, 
              reg_minmax,
              scheme='volume-stoichiometry',
              isol_es=isol_es
              )
    assert True # TODO: modify this to test actual condition
    