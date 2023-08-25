"""
Jobs to create training data for ML potentials
"""
from __future__ import annotations


from jobflow import Flow, Response, job, Maker
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np


@job
def generate_random_displacement(
        structure: Structure
):
    random_displacements = []
    ase_structure = AseAtomsAdaptor.get_atoms(structure)
    for seed in np.random.permutation(100)[:1]:
        ase_structure.rattle(seed=seed)
        random_displacements.append(AseAtomsAdaptor.get_structure(ase_structure))
    return random_displacements
