"""Flows to create training data for ML potentials."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atomate2.vasp.jobs.base import BaseVaspMaker
    from atomate2.vasp.sets.base import VaspInputGenerator
    from emmet.core.math import Matrix3D
    from pymatgen.core.structure import Species, Structure
from atomate2.common.jobs.phonons import (
    PhononDisplacementMaker,
    run_phonon_displacements,
)
from atomate2.vasp.jobs.core import StaticMaker
from atomate2.vasp.powerups import (
    update_user_incar_settings,
)
from atomate2.vasp.sets.core import StaticSetGenerator
from jobflow import Flow, Maker
from phonopy.structure.cells import get_supercell
from pymatgen.core import Molecule, Site
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

from autoplex.data.jobs import generate_randomized_structures

__all__ = ["RandomStructuresDataGenerator", "IsoAtomMaker"]


@dataclass
class APPhononDisplacementMaker(PhononDisplacementMaker):
    """Adapted phonon displacement maker for static calculation.

    The input set is for a static run same as PhononDisplacementMaker.
    Only difference is Spin polarization is switched off and Gaussian smearing is used

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .VaspInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_vasp_input_set`.
    copy_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_vasp_outputs`.
    run_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_vasp`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDoc.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_kpoints_settings={"reciprocal_density": 100},
            user_incar_settings={
                "IBRION": 2,
                "ISPIN": 1,
                "ISMEAR": 0,
                "ISIF": 3,
                "ENCUT": 700,
                "EDIFF": 1e-7,
                "LAECHG": False,
                "LREAL": False,
                "ALGO": "Normal",
                "NSW": 0,
                "LCHARG": False,
            },
            auto_ispin=False,
        )
    )


@dataclass
class IsoAtomStaticMaker(StaticMaker):
    """
    Maker to create Isolated atoms static jobs.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .VaspInputGenerator
        A generator used to make the input set.
    write_input_set_kwargs : dict
        Keyword arguments that will get passed to :obj:`.write_vasp_input_set`.
    copy_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.copy_vasp_outputs`.
    run_vasp_kwargs : dict
        Keyword arguments that will get passed to :obj:`.run_vasp`.
    task_document_kwargs : dict
        Keyword arguments that will get passed to :obj:`.TaskDoc.from_directory`.
    stop_children_kwargs : dict
        Keyword arguments that will get passed to :obj:`.should_stop_children`.
    write_additional_data : dict
        Additional data to write to the current directory. Given as a dict of
        {filename: data}. Note that if using FireWorks, dictionary keys cannot contain
        the "." character which is typically used to denote file extensions. To avoid
        this, use the ":" character, which will automatically be converted to ".". E.g.
        ``{"my_file:txt": "contents of the file"}``.
    """

    name: str = "static"
    input_set_generator: VaspInputGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_kpoints_settings={"reciprocal_density": 1},
            user_incar_settings={
                "ISPIN": 1,
                "LAECHG": False,
                "ISMEAR": 0,
            },
        )
    )


@dataclass
class RandomStructuresDataGenerator(Maker):
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
    uc: bool.
        If True, will use the unit cells of initial randomly displaced
        structures and add phonon static computation jobs to the flow
    """

    name: str = "RandomStruturesDataGeneratorForML"
    phonon_displacement_maker: BaseVaspMaker = field(
        default_factory=PhononDisplacementMaker
    )
    code: str = "vasp"
    n_struct: int = 1
    uc: bool = False

    def make(
        self,
        structure: Structure,
        mp_id: str,
        supercell_matrix: Matrix3D | None = None,
    ):
        """
        Make flow to generate the reference DFT data base.

        Parameters
        ----------
        structure :
            Pymatgen structures drawn from the Materials Project.
        mp_id: str
            Materials Project IDs
        supercell_matrix: Matrix3D.
            Matrix for obtaining the supercell
        """
        jobs = []  # initializing empty job list
        outputs = []

        if supercell_matrix is None:
            supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        supercell = get_supercell(
            unitcell=get_phonopy_structure(structure),
            supercell_matrix=supercell_matrix,
        )
        random_rattle_sc = generate_randomized_structures(
            structure=get_pmg_structure(supercell), n_struct=self.n_struct
        )
        jobs.append(random_rattle_sc)
        # perform the phonon displaced calculations for randomized displaced structures
        # structure is only needed to keep track of the original structure
        vasp_random_sc_displacement_calcs = run_phonon_displacements(
            displacements=random_rattle_sc.output,  # pylint: disable=E1101
            structure=structure,
            supercell_matrix=None,
            phonon_maker=self.phonon_displacement_maker,
        )
        vasp_random_sc_displacement_calcs = update_user_incar_settings(
            vasp_random_sc_displacement_calcs,
            {"NPAR": 4, "ISPIN": 1, "LAECHG": False, "ISMEAR": 0},
        )
        jobs.append(vasp_random_sc_displacement_calcs)
        outputs.append(vasp_random_sc_displacement_calcs.output["dirs"])

        if self.uc is True:
            random_rattle = generate_randomized_structures(
                structure=structure, n_struct=self.n_struct
            )
            jobs.append(random_rattle)
            vasp_random_displacement_calcs = run_phonon_displacements(
                displacements=random_rattle.output,  # pylint: disable=E1101
                structure=structure,
                supercell_matrix=None,
                phonon_maker=self.phonon_displacement_maker,
            )
            vasp_random_displacement_calcs = update_user_incar_settings(
                vasp_random_displacement_calcs,
                {"NPAR": 4, "ISPIN": 1, "LAECHG": False, "ISMEAR": 0},
            )
            jobs.append(vasp_random_displacement_calcs)
            outputs.append(vasp_random_displacement_calcs.output["dirs"])

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

    def make(self, all_species: list[Species]):
        """
        Make a flow to calculate the isolated atom's energy.

        Parameters
        ----------
        all_species : List of Species
            list of pymatgen specie object.
        """
        jobs = []
        isoatoms = []
        for species in all_species:
            site = Site(species=species, coords=[0, 0, 0])
            mol = Molecule.from_sites([site])
            iso_atom = mol.get_boxed_structure(a=20, b=20, c=20)
            isoatom_calcs = StaticMaker(
                name=str(species) + "-statisoatom",
                input_set_generator=StaticSetGenerator(
                    user_kpoints_settings={"grid_density": 1},
                ),
            ).make(iso_atom)

            isoatom_calcs = update_user_incar_settings(
                isoatom_calcs,
                {"NPAR": 4, "ISPIN": 1, "LAECHG": False, "ISMEAR": 0},
            )
            jobs.append(isoatom_calcs)
            isoatoms.append(isoatom_calcs.output.output.energy_per_atom)
        # create a flow including all jobs
        return Flow(jobs, isoatoms)
