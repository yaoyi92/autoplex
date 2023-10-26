"""Jobs to create training data for ML potentials."""
from __future__ import annotations

import numpy as np
from atomate2.vasp.flows.phonons import PhononMaker as DFTPhononMaker
from jobflow import Flow, Response, job
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor


@job
def generate_randomized_structures(
    structure: Structure,
    n_struct: int,
):
    """
    Take in a structure object and generates randomly displaced structure.

    Parameters
    ----------
    structure : Structure.
        Pymatgen structures object.
    n_struct : int.
        Total number of randomly displaced structures to be generated.

    Returns
    -------
    Response.output.
        Randomly displaced structures.
    """
    random_rattled = []
    ase_structure = AseAtomsAdaptor.get_atoms(structure)
    for seed in np.random.permutation(100)[:n_struct]:
        ase_structure.rattle(seed=seed)
        random_rattled.append(AseAtomsAdaptor.get_structure(ase_structure))
    return random_rattled


@job
def phonon_maker_random_structures(
    rattled_structures, displacements, symprec, phonon_displacement_maker
):
    """
    Set up phonon computations of the randomly displaced structure.

    Parameters
    ----------
    rattled_structures : List[Structure].
        list of randomly displaced pymatgen structures objects.
    displacements : float.
        displacement distance for phonons.
    symprec : float
        Symmetry precision to use in the
        reduction of symmetry to find the primitive/conventional cell
        (use_primitive_standard_structure, use_conventional_standard_structure)
        and to handle all symmetry-related tasks in phonopy
    phonon_displacement_maker : .BaseVaspMaker or None
        Maker used to compute the forces for a supercell.

    Returns
    -------
    list.
        List of phonon jobs.
    """
    jobs = []
    outputs = []
    for rand_struc in rattled_structures:
        for displacement in displacements:
            random_phonon_maker = DFTPhononMaker(
                symprec=symprec,
                phonon_displacement_maker=phonon_displacement_maker,
                born_maker=None,
                bulk_relax_maker=None,
                min_length=8,
                displacement=displacement,
            ).make(structure=rand_struc)
            jobs.append(random_phonon_maker)
            outputs.append(random_phonon_maker.output)
    flow = Flow(jobs, outputs)
    return Response(replace=flow)
