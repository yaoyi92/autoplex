from autoplex.data.phonons.utils import reduce_supercell_size
from jobflow import job
from typing import TYPE_CHECKING
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)
from pymatgen.core.structure import Structure
if TYPE_CHECKING:
    from atomate2.forcefields.jobs import (
        ForceFieldRelaxMaker,
        ForceFieldStaticMaker,
    )
    from atomate2.vasp.jobs.phonons import PhononDisplacementMaker



@job
def reduce_supercell_size_job(structure: Structure,
        min_length: float = 18,
        max_length: float = 22,
        fallback_min_length: float = 12,
        min_atoms: int = 100,
        max_atoms: int = 500,
        step_size: float = 1,):
    return reduce_supercell_size(structure=structure, min_length=min_length,max_length=max_length, fallback_min_length=fallback_min_length, min_atoms=min_atoms,
                                 max_atoms=max_atoms,step_size=step_size)
