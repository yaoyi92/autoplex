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
    # leaving this here and adding the duplicate in common to avoid the respective unit tests from failing
    structure: Structure,
    n_struct: int,
    cell_factor_sequence: list[float] | None = None,
    std_dev: float = 0.01,
):
    """
    Take in a pymatgen Structure object and generates randomly displaced structures.

    Parameters
    ----------
    structure : Structure.
        Pymatgen structures object.
    n_struct : int.
        Total number of randomly displaced structures to be generated.
    cell_factor_sequence: list[float]
        list of factors to resize cell parameters.
    std_dev: float
        Standard deviation std_dev for normal distribution to draw numbers from to generate the rattled structures.

    Returns
    -------
    Response.output.
        Randomly displaced structures.
    """
    random_rattled = []
    if cell_factor_sequence is None:
        cell_factor_sequence = [0.975, 1.0, 1.025, 1.05]
    for cell_factor in cell_factor_sequence:
        ase_structure = AseAtomsAdaptor.get_atoms(structure)
        ase_structure.set_cell(ase_structure.get_cell() * cell_factor, scale_atoms=True)
        for seed in np.random.permutation(100000)[:n_struct]:
            ase_structure.rattle(seed=seed, stdev=std_dev)
            random_rattled.append(AseAtomsAdaptor.get_structure(ase_structure))
    return random_rattled
