"""
Jobs to create training data for ML potentials
"""
from __future__ import annotations


from jobflow import Flow, Response, job, Maker
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from atomate2.vasp.flows.phonons import PhononMaker as DFTPhononMaker
import numpy as np


@job
def generate_randomized_structures(
        structure: Structure,
        n_struc: int,
):
    random_rattled = []
    ase_structure = AseAtomsAdaptor.get_atoms(structure)
    for seed in np.random.permutation(100)[:n_struc]:
        ase_structure.rattle(seed=seed)
        random_rattled.append(AseAtomsAdaptor.get_structure(ase_structure))
    return random_rattled

@job
def phonon_maker_random_structures(
        rattled_structures,
        displacements,
        symprec,
        phonon_displacement_maker
):
    jobs = []
    outputs = []
    for rand_struc in rattled_structures:
        for displacement in displacements:
            random_phonon_maker = DFTPhononMaker(symprec=symprec,
                                                 phonon_displacement_maker=phonon_displacement_maker,
                                                 born_maker=None, bulk_relax_maker=None,
                                                 min_length=8, displacement=displacement).make(structure=rand_struc)
            jobs.append(random_phonon_maker)
            outputs.append(random_phonon_maker.output)
    flow = Flow(jobs, outputs)
    return Response(replace=flow)
