"""Flows to create training data for ML potentials."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from atomate2.vasp.jobs.base import BaseVaspMaker
    from emmet.core.math import Matrix3D

from atomate2.common.jobs.phonons import (
    PhononDisplacementMaker,
    run_phonon_displacements,
)
from atomate2.vasp.jobs.core import StaticMaker
from atomate2.vasp.sets.core import StaticSetGenerator
from jobflow import Flow, Maker
from phonopy import Phonopy
from pymatgen.core.structure import Species, Structure
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

from autoplex.data.jobs import generate_randomized_structures

__all__ = ["RandomStruturesDataGenerator", "IsoAtomMaker"]


@dataclass
class RandomStruturesDataGenerator(Maker):
    """
    Maker to generate DFT data based on random displacements for ML potential fitting.

    1. Randomizes Structures (with and without supercell).
    2. Performs DFT calculations.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    phonon_displacement_maker : .BaseVaspMaker or None
        Maker used to compute the forces for a supercell.
    code: str
        determines the dft code. currently only vasp is implemented.
        This keyword might enable the implementation of other codes
        in the future
    n_struct: int.
        The total number of randomly displaced structures to be generated.
    sc: bool.
        If True, will generate supercells of initial randomly displaced
        structures and add phonon computation jobs to the flow
    """

    name: str = "RandomStruturesDataGeneratorForML"
    phonon_displacement_maker: BaseVaspMaker = field(
        default_factory=PhononDisplacementMaker
    )
    code: str = "vasp"
    n_struct: int = 1
    sc: bool = False

    def make(
        self,
        structure: Structure,
        mp_id: str,
        prev_vasp_dir: str | Path | None = None,
        supercell_matrix: Matrix3D
        | None = None,  # with a simpler static vasp method this will be redundant
    ):
        """
        Make flow to generate the reference DFT data base.

        Parameters
        ----------
        structure :
            Pymatgen structures drawn from the Materials Project.
        mp_id: str
            Materials Project IDs
        prev_vasp_dir: str or Path or None
             A previous vasp calculation directory to use for copying outputs.
        supercell_matrix: Matrix3D.
            Matrix for obtaining the supercell
        """
        # TODO: clean up unused arguments: is prev_vasp_dir needed?

        jobs = []  # initializing empty job list
        outputs = []

        random_rattle = generate_randomized_structures(
            structure=structure, n_struct=self.n_struct
        )
        jobs.append(random_rattle)
        # perform the phonon displaced calculations for randomized displaced structures
        # could be replaced with a simple static_vasp method
        # structure is only needed to keep track of the original structure
        vasp_random_displacement_calcs = run_phonon_displacements(
            displacements=random_rattle.output,  # pylint: disable=E1101
            structure=structure,
            supercell_matrix=supercell_matrix,
            phonon_maker=self.phonon_displacement_maker,
        )
        jobs.append(vasp_random_displacement_calcs)
        outputs.append(vasp_random_displacement_calcs.output["dirs"])

        if self.sc is True:
            supercell = Phonopy(unitcell=get_phonopy_structure(structure)).supercell
            random_rattle_sc = generate_randomized_structures(
                structure=get_pmg_structure(supercell), n_struct=self.n_struct
            )
            jobs.append(random_rattle_sc)
            # could be replaced with a simple static_vasp method
            vasp_random_sc_displacement_calcs = run_phonon_displacements(
                displacements=random_rattle_sc.output,  # pylint: disable=E1101
                structure=structure,
                supercell_matrix=supercell_matrix,
                phonon_maker=self.phonon_displacement_maker,
            )
            # line 126 structure is only needed to keep track of the original structure
            jobs.append(vasp_random_sc_displacement_calcs)
            outputs.append(vasp_random_sc_displacement_calcs.output["dirs"])

        # create a flow including all jobs
        return Flow(jobs, outputs)


@dataclass
class IsoAtomMaker(Maker):
    """
    Maker to generate DFT data for ML potential fitting from isolated atoms.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    """

    name: str = "IsolatedAtomEnergyMaker"

    def make(self, species: Species):
        """
        Make a flow to calculate the isolated atom's energy.

        Parameters
        ----------
        species : Species
            pymatgen specie object.
        """
        jobs = []
        iso_atom = Structure(
            lattice=[[20, 0, 0], [0, 20, 0], [0, 0, 20]],  # TODO replace with boxed
            species=[species],
            coords=[[0, 0, 0]],
        )
        isoatom_calcs = StaticMaker(
            name=str(species) + "-statisoatom",
            input_set_generator=StaticSetGenerator(
                user_kpoints_settings={"grid_density": 1},
            ),
        ).make(iso_atom)
        jobs.append(isoatom_calcs)
        # create a flow including all jobs
        return Flow(jobs, isoatom_calcs.output.output.energy_per_atom)
