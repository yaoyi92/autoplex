"""
Flows consisting of jobs to create training data for ML potentials
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from jobflow import Flow, Maker, OutputReference, job
from pymatgen.core.structure import Structure
from autoplex.data.jobs import generate_randomized_structures, phonon_maker_random_structures
from atomate2.vasp.sets.core import StaticSetGenerator
from emmet.core.math import Matrix3D
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import StaticMaker
from atomate2.common.jobs.phonons import (
    PhononDisplacementMaker,
    run_phonon_displacements as run_static,
)

__all__ = ["DataGenerator", "IsoAtomMaker"]


@dataclass
class DataGenerator(Maker):
    """
    Maker to create DFT data for ML potential fitting
    1. Fetch Data from Materials Project and other databases (other: work in progress)
    + Perform DFT calculations (at the current point these are also used for Phonon calculations (that part
    shall be independent in the future.)

--Note:
    All phonon-related code parts have been copied from atomate2/vasp/flows/phonons.py

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.

    """

    name: str = "DataGenerationML"
    phonon_displacement_maker: BaseVaspMaker = field(default_factory=PhononDisplacementMaker)
    code: str = "vasp"
    n_struc: int = 1
    displacements: list[float] = field(default=0.1)
    symprec: float = 0.01

    def make(
            self,
            structure: Structure,
            mpid: int,
            prev_vasp_dir: str | Path | None = None,
            total_dft_energy_per_formula_unit: float | None = None,
            supercell_matrix: Matrix3D | None = None,
    ):
        """
        Make flow to generate the data base.

        Parameters
        ----------
        structure :
            Pymatgen structures drawn from the Materials Project.
        ml_dir : str or Path or None
            ML directory to use for copying inputs/outputs.(hab noch keine Ahnung)
        supercell_matrix: Matrix of the SC
        mpids: Materials Project IDs
        """
        jobs = []  # initializing empty job list
        outputs = []

        random_rattle = generate_randomized_structures(structure=structure, n_struc=self.n_struc)
        jobs.append(random_rattle)

        # perform the phonon displaced calculations for randomized displaced structures

        random_phonon_maker = phonon_maker_random_structures(rattled_structures=random_rattle.output,
                                                             displacements=self.displacements, symprec=self.symprec,
                                                             phonon_displacement_maker=self.phonon_displacement_maker)
        jobs.append(random_phonon_maker)
        outputs.append(random_phonon_maker.output.jobdirs.displacements_job_dirs)

        vasp_random_displacement_calcs = run_static( # could be replaced with a simple static_vasp method
            displacements=random_rattle.output,
            structure=structure,  # strucure is only needed to keep track of the original structure
            supercell_matrix=supercell_matrix,
            phonon_maker=self.phonon_displacement_maker,
        )
        jobs.append(vasp_random_displacement_calcs)
        outputs.append(vasp_random_displacement_calcs.output['dirs'])

        # create a flow including all jobs
        flow = Flow(jobs, outputs)
        return flow


@dataclass
class IsoAtomMaker(Maker):
    """
    Class to calculate isolated atoms energy for GAP fit
    """
    name: str = "IsolatedAtomEnergyMaker"

    def make(self, species):
        """
        Returns a VASP job to calculate the isolated atoms energy.
        """
        jobs = []
        iso_atom = Structure(
            lattice=[[20, 0, 0], [0, 20, 0], [0, 0, 20]],
            species=[species],
            coords=[[0, 0, 0]],
        )
        isoatom_calcs = StaticMaker(name=str(species) + "-statisoatom", input_set_generator=StaticSetGenerator(
            user_kpoints_settings={"grid_density": 1}, )).make(iso_atom)
        jobs.append(isoatom_calcs)
        # create a flow including all jobs
        flow = Flow(jobs, isoatom_calcs.output.output.energy_per_atom)
        return flow
