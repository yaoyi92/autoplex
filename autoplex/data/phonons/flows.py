"""Flows to create training data for ML potentials."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atomate2.vasp.jobs.base import BaseVaspMaker
    from atomate2.vasp.sets.base import VaspInputGenerator
    from emmet.core.math import Matrix3D
    from pymatgen.core.structure import Species, Structure
from atomate2.common.jobs.phonons import run_phonon_displacements
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.flows.phonons import PhononMaker
from atomate2.vasp.jobs.core import StaticMaker, TightRelaxMaker
from atomate2.vasp.jobs.phonons import PhononDisplacementMaker
from atomate2.vasp.sets.core import StaticSetGenerator, TightRelaxSetGenerator
from jobflow import Flow, Maker
from phonopy.structure.cells import get_supercell
from pymatgen.core import Molecule, Site
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

from autoplex.data.common.jobs import generate_randomized_structures

__all__ = [
    "DFTPhononMaker",
    "IsoAtomMaker",
    "IsoAtomStaticMaker",
    "RandomStructuresDataGenerator",
    "TightDFTStaticMaker",
]


@dataclass
class TightDFTStaticMaker(PhononDisplacementMaker):
    """Adapted phonon displacement maker for static calculation.

    The input set used is same as PhononDisplacementMaker.
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

    run_vasp_kwargs: dict = field(default_factory=lambda: {"handlers": ()})

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
                "ALGO": "Fast",
                "NSW": 0,
                "LCHARG": False,
                "SIGMA": 0.05,
            },
            auto_ispin=False,
        )
    )


@dataclass
class DFTPhononMaker(PhononMaker):
    """
    Adapted PhononMaker to calculate harmonic phonons with VASP and Phonopy.

    The input set used is same as PhononMaker from atomate2.
    Only difference is Spin polarization is switched off and Gaussian smearing is used

    Parameters
    ----------
    name : str = "phonon"
        Name of the flows produced by this maker.
    sym_reduce : bool = True
        Whether to reduce the number of deformations using symmetry.
    symprec : float = 1e-4
        Symmetry precision to use in the
        reduction of symmetry to find the primitive/conventional cell
        (use_primitive_standard_structure, use_conventional_standard_structure)
        and to handle all symmetry-related tasks in phonopy
    displacement: float = 0.01
        displacement distance for phonons
    min_length: float = 20.0
        min length of the supercell that will be built
    prefer_90_degrees: bool = True
        if set to True, supercell algorithm will first try to find a supercell
        with 3 90 degree angles
    get_supercell_size_kwargs: dict = {}
        kwargs that will be passed to get_supercell_size to determine supercell size
    use_symmetrized_structure: str or None = None
        allowed strings: "primitive", "conventional", None

        - "primitive" will enforce to start the phonon computation
          from the primitive standard structure
          according to Setyawan, W., & Curtarolo, S. (2010).
          High-throughput electronic band structure calculations:
          Challenges and tools. Computational Materials Science,
          49(2), 299-312. doi:10.1016/j.commatsci.2010.05.010.
          This makes it possible to use certain k-path definitions
          with this workflow. Otherwise, we must rely on seekpath
        - "conventional" will enforce to start the phonon computation
          from the conventional standard structure
          according to Setyawan, W., & Curtarolo, S. (2010).
          High-throughput electronic band structure calculations:
          Challenges and tools. Computational Materials Science,
          49(2), 299-312. doi:10.1016/j.commatsci.2010.05.010.
          We will however use seekpath and primitive structures
          as determined by from phonopy to compute the phonon band structure
    bulk_relax_maker : .BaseVaspMaker or None
        A maker to perform a tight relaxation on the bulk.
        Set to ``None`` to skip the
        bulk relaxation
    static_energy_maker : .BaseVaspMaker or None
        A maker to perform the computation of the DFT energy on the bulk.
        Set to ``None`` to skip the
        static energy computation
    born_maker: .BaseVaspMaker or None
        Maker to compute the BORN charges.
    phonon_displacement_maker : .BaseVaspMaker or None
        Maker used to compute the forces for a supercell.
    generate_frequencies_eigenvectors_kwargs : dict
        Keyword arguments passed to :obj:`generate_frequencies_eigenvectors`.
    create_thermal_displacements: bool
        Arg that determines if thermal_displacement_matrices are computed
    kpath_scheme: str = "seekpath"
        scheme to generate kpoints. Please be aware that
        you can only use seekpath with any kind of cell
        Otherwise, please use the standard primitive structure
        Available schemes are:
        "seekpath", "hinuma", "setyawan_curtarolo", "latimer_munro".
        "seekpath" and "hinuma" are the same definition but
        seekpath can be used with any kind of unit cell as
        it relies on phonopy to handle the relationship
        to the primitive cell and not pymatgen
    code: str = "vasp"
        determines the DFT code. currently only vasp is implemented.
        This keyword might enable the implementation of other codes
        in the future
    store_force_constants: bool
        if True, force constants will be stored
    """

    name: str = "phonon"
    sym_reduce: bool = True
    symprec: float = 1e-4
    displacement: float = 0.01
    min_length: float | None = 20.0
    prefer_90_degrees: bool = True
    get_supercell_size_kwargs: dict = field(default_factory=dict)
    use_symmetrized_structure: str | None = None
    bulk_relax_maker: BaseVaspMaker | None = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(
            TightRelaxMaker(
                input_set_generator=TightRelaxSetGenerator(
                    user_incar_settings={"ISPIN": 1, "LAECHG": False, "ISMEAR": 0}
                )
            )
        ),
    )
    static_energy_maker: BaseVaspMaker | None = field(
        default_factory=lambda: StaticMaker(
            input_set_generator=StaticSetGenerator(
                auto_ispin=False,
                user_incar_settings={"ISPIN": 1, "LAECHG": False, "ISMEAR": 0},
            )
        )
    )
    phonon_displacement_maker: BaseVaspMaker = field(
        default_factory=TightDFTStaticMaker
    )


@dataclass
class IsoAtomStaticMaker(StaticMaker):
    """
    Maker to create Isolated atoms static (VASP) jobs.

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
    Maker to generate DFT labelled training data for ML potential fitting based on random atomic displacements.

    This Maker performs the two following steps:
    1. Generates supercells from the provided structure and randomly displaces the atomic positions using ase rattle.
    (randomized unit cells can be generated additionally).
    2. Performs the static DFT (VASP) calculations on the randomized cells.

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
    rattle_std: float.
        Rattle amplitude (standard deviation in normal distribution).
        Default=0.01.
    """

    name: str = "RandomStruturesDataGeneratorForML"
    phonon_displacement_maker: BaseVaspMaker = field(
        default_factory=TightDFTStaticMaker
    )
    code: str = "vasp"
    n_structures: int = 1
    uc: bool = False
    rattle_std: float = 0.01

    def make(
        self,
        structure: Structure,
        mp_id: str,
        supercell_matrix: Matrix3D | None = None,
        volume_custom_scale_factors: list[float] | None = None,
    ):
        """
        Make a flow to generate rattled structures reference DFT data.

        Parameters
        ----------
        structure :
            Pymatgen structures drawn from the Materials Project.
        mp_id: str
            Materials Project IDs
        supercell_matrix: Matrix3D.
            Matrix for obtaining the supercell
        volume_custom_scale_factors : list[float]
            Specify explicit scale factors (if range is not specified).
            If None, will default to [0.90, 0.95, 0.98, 0.99, 1.01, 1.02, 1.05, 1.10].
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
            structure=get_pmg_structure(supercell),
            n_structures=self.n_structures,
            volume_custom_scale_factors=volume_custom_scale_factors,
            rattle_std=self.rattle_std,
        )
        jobs.append(random_rattle_sc)
        # perform the phonon displaced calculations for randomized displaced structures.
        #  The original structure is only needed to keep track of initial structure.
        vasp_random_sc_displacement_calcs = run_phonon_displacements(
            displacements=random_rattle_sc.output,  # pylint: disable=E1101
            structure=structure,
            supercell_matrix=None,
            phonon_maker=self.phonon_displacement_maker,
        )

        jobs.append(vasp_random_sc_displacement_calcs)
        outputs.append(vasp_random_sc_displacement_calcs.output["dirs"])

        if self.uc is True:
            random_rattle = generate_randomized_structures(
                structure=structure,
                n_structures=self.n_structures,
                volume_custom_scale_factors=volume_custom_scale_factors,
                rattle_std=self.rattle_std,
            )
            jobs.append(random_rattle)
            vasp_random_displacement_calcs = run_phonon_displacements(
                displacements=random_rattle.output,  # pylint: disable=E1101
                structure=structure,
                supercell_matrix=None,
                phonon_maker=self.phonon_displacement_maker,
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
        isoatoms_energy = []
        isoatoms_dirs = []
        for species in all_species:
            site = Site(species=species, coords=[0, 0, 0])
            mol = Molecule.from_sites([site])
            iso_atom = mol.get_boxed_structure(a=20, b=20, c=20)
            isoatom_calcs = IsoAtomStaticMaker(
                name=str(species) + "-statisoatom",
                input_set_generator=StaticSetGenerator(
                    user_kpoints_settings={"grid_density": 1},
                ),
            ).make(iso_atom)

            jobs.append(isoatom_calcs)
            isoatoms_energy.append(isoatom_calcs.output.output.energy_per_atom)
            isoatoms_dirs.append(isoatom_calcs.output.dir_name)

        # create a flow including all jobs
        return Flow(jobs, {"energies": isoatoms_energy, "dirs": isoatoms_dirs})
