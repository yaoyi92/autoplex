"""Jobs to create training data for ML potentials."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from jobflow import job

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor


@job
def generate_randomized_structures(
    structure: Structure, n_struct: int, cell_factor: float = 1.0, std_dev: float = 0.01
):
    """
    Take in a structure object and generates randomly displaced structure.

    Parameters
    ----------
    structure : Structure.
        Pymatgen structures object.
    n_struct : int.
        Total number of randomly displaced structures to be generated.
    cell_factor: float
        factor to resize cell parameters.
    std_dev: float
        Standard deviation std_dev for normal distribution to draw numbers from to generate the rattled structures.

    Returns
    -------
    Response.output.
        Randomly displaced structures.
    """
    random_rattled = []
    ase_structure = AseAtomsAdaptor.get_atoms(structure)
    ase_structure.set_cell(ase_structure.get_cell() * cell_factor, scale_atoms=True)
    for seed in np.random.permutation(100000)[:n_struct]:
        ase_structure.rattle(seed=seed, stdev=std_dev)
        random_rattled.append(AseAtomsAdaptor.get_structure(ase_structure))
    return random_rattled
