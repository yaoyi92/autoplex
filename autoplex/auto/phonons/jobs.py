"""General AutoPLEX automation jobs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atomate2.forcefields.flows.phonons import PhononMaker
from atomate2.forcefields.jobs import (
    ForceFieldRelaxMaker,
    ForceFieldStaticMaker,
    GAPRelaxMaker,
    GAPStaticMaker,
)
from atomate2.vasp.flows.phonons import PhononMaker as DFTPhononMaker
from atomate2.vasp.powerups import (
    update_user_incar_settings,
)
from jobflow import Flow, Response, job

if TYPE_CHECKING:
    from emmet.core.math import Matrix3D
    from pymatgen.core.structure import Structure

from autoplex.data.phonons.flows import IsoAtomMaker, RandomStructuresDataGenerator


@dataclass
class MLPhononMaker(PhononMaker):
    """
    Maker to calculate harmonic phonons with a force field.

    Calculate the harmonic phonons of a material. Initially, a tight structural
    relaxation is performed to obtain a structure without forces on the atoms.
    Subsequently, supercells with one displaced atom are generated and accurate
    forces are computed for these structures. With the help of phonopy, these
    forces are then converted into a dynamical matrix. To correct for polarization
    effects, a correction of the dynamical matrix based on BORN charges can
    be performed. The BORN charges can be supplied manually.
    Finally, phonon densities of states, phonon band structures
    and thermodynamic properties are computed.

    .. Note::
        It is heavily recommended to symmetrize the structure before passing it to
        this flow. Otherwise, a different space group might be detected and too
        many displacement calculations will be generated.
        It is recommended to check the convergence parameters here and
        adjust them if necessary. The default might not be strict enough
        for your specific case.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    sym_reduce : bool
        Whether to reduce the number of deformations using symmetry.
    symprec : float
        Symmetry precision to use in the
        reduction of symmetry to find the primitive/conventional cell
        (use_primitive_standard_structure, use_conventional_standard_structure)
        and to handle all symmetry-related tasks in phonopy
    displacement: float
        displacement distance for phonons
    min_length: float
        min length of the supercell that will be built
    prefer_90_degrees: bool
        if set to True, supercell algorithm will first try to find a supercell
        with 3 90 degree angles
    get_supercell_size_kwargs: dict
        kwargs that will be passed to get_supercell_size to determine supercell size
    use_symmetrized_structure: str
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
    bulk_relax_maker : .ForceFieldRelaxMaker or None
        A maker to perform a tight relaxation on the bulk.
        Set to ``None`` to skip the
        bulk relaxation
    static_energy_maker : .ForceFieldStaticMaker or None
        A maker to perform the computation of the DFT energy on the bulk.
        Set to ``None`` to skip the
        static energy computation
    phonon_displacement_maker : .ForceFieldStaticMaker or None
        Maker used to compute the forces for a supercell.
    generate_frequencies_eigenvectors_kwargs : dict
        Keyword arguments passed to :obj:`generate_frequencies_eigenvectors`.
    create_thermal_displacements: bool
        Arg that determines if thermal_displacement_matrices are computed
    kpath_scheme: str
        scheme to generate kpoints. Please be aware that
        you can only use seekpath with any kind of cell
        Otherwise, please use the standard primitive structure
        Available schemes are:
        "seekpath", "hinuma", "setyawan_curtarolo", "latimer_munro".
        "seekpath" and "hinuma" are the same definition but
        seekpath can be used with any kind of unit cell as
        it relies on phonopy to handle the relationship
        to the primitive cell and not pymatgen
    code: str
        determines the DFT code. currently only vasp is implemented.
        This keyword might enable the implementation of other codes
        in the future
    store_force_constants: bool
        if True, force constants will be stored
    """

    min_length: float | None = 20.0
    bulk_relax_maker: ForceFieldRelaxMaker | None = field(
        default_factory=lambda: GAPRelaxMaker(
            relax_cell=True, relax_kwargs={"interval": 500}
        )
    )
    phonon_displacement_maker: ForceFieldStaticMaker | None = field(
        default_factory=lambda: GAPStaticMaker()
    )
    static_energy_maker: ForceFieldStaticMaker | None = field(
        default_factory=lambda: GAPStaticMaker()
    )
    store_force_constants: bool = False
    generate_frequencies_eigenvectors_kwargs: dict = field(
        default_factory=lambda: {"units": "THz"}
    )
    relax_maker_kwargs: dict = field(default_factory=dict)
    static_maker_kwargs: dict = field(default_factory=dict)

    @job
    def make_from_ml_model(self, structure, ml_model, **make_kwargs):
        """
        Maker for GAP phonon jobs.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure. Please start with a structure
            that is nearly fully optimized as the internal optimizers
            have very strict settings!
        ml_model : str
            Complete path to gapfit.xml file including file name.
        make_kwargs :
            Keyword arguments for the PhononMaker.

        Returns
        -------
        PhononMaker jobs.

        """
        ml_model = ml_model + "/gap_file.xml"
        if self.bulk_relax_maker is not None:
            br = self.bulk_relax_maker
            self.bulk_relax_maker = br.update_kwargs(
                update={
                    "potential_param_file_name": ml_model,
                    **self.relax_maker_kwargs,
                }
            )
        if self.phonon_displacement_maker is not None:
            ph_disp = self.phonon_displacement_maker
            self.phonon_displacement_maker = ph_disp.update_kwargs(
                update={
                    "potential_param_file_name": ml_model,
                    **self.static_maker_kwargs,
                }
            )
        if self.static_energy_maker is not None:
            stat_en = self.static_energy_maker
            self.static_energy_maker = stat_en.update_kwargs(
                update={
                    "potential_param_file_name": ml_model,
                    **self.static_maker_kwargs,
                }
            )

        flow = self.make(structure=structure, **make_kwargs)
        return Response(replace=flow, output=flow.output)


@job
def dft_phonopy_gen_data(
    structure: Structure, displacements, symprec, phonon_displacement_maker, min_length
):
    """
    Job to generate DFT reference database using phonopy to be used for fitting ML potentials.

    Parameters
    ----------
    structure: Structure
        pymatgen Structure object
    phonon_displacement_maker : .BaseVaspMaker or None
        Maker used to compute the forces for a supercell.
    displacements: list[float]
        list of phonon displacement
    min_length: float
        min length of the supercell that will be built
    symprec : float
        Symmetry precision to use in the
        reduction of symmetry to find the primitive/conventional cell
        (use_primitive_standard_structure, use_conventional_standard_structure)
        and to handle all symmetry-related tasks in phonopy
    """
    jobs = []
    dft_phonons_output = {}
    dft_phonons_dir_output = []

    for displacement in displacements:
        dft_phonons = DFTPhononMaker(
            symprec=symprec,
            phonon_displacement_maker=phonon_displacement_maker,
            born_maker=None,
            displacement=displacement,
            min_length=min_length,
        ).make(structure=structure)
        dft_phonons = update_user_incar_settings(
            dft_phonons, {"NPAR": 4, "ISPIN": 1, "LAECHG": False, "ISMEAR": 0}
        )
        jobs.append(dft_phonons)
        dft_phonons_output[
            f"{displacement}".replace(".", "")  # key must not contain '.'
        ] = dft_phonons.output
        dft_phonons_dir_output.append(dft_phonons.output.jobdirs.displacements_job_dirs)

    flow = Flow(jobs, {"dirs": dft_phonons_dir_output, "data": dft_phonons_output})
    return Response(replace=flow)


@job
def dft_random_gen_data(
    structure: Structure,
    mp_id,
    phonon_displacement_maker,
    n_struct: int = 1,
    uc: bool = False,
    cell_factor: float = 1.0,
    std_dev: float = 0.01,
    supercell_matrix: Matrix3D | None = None,
):
    """
    Job to generate random structured DFT reference database to be used for fitting ML potentials.

    Parameters
    ----------
    structure: Structure
        pymatgen Structure object
    phonon_displacement_maker : .BaseVaspMaker or None
        Maker used to compute the forces for a supercell.
    mp_id:
        materials project id
    n_struct: int.
        The total number of randomly displaced structures to be generated.
    uc: bool.
        If True, will generate randomly distorted structures (unitcells)
        and add static computation jobs to the flow.
    cell_factor: float
        factor to resize cell parameters.
    std_dev: float
        Standard deviation std_dev for normal distribution to draw numbers from to generate the rattled structures.
    supercell_matrix: Matrix3D or None
        The matrix to construct the supercell.
    """
    jobs = []
    random_datagen = RandomStructuresDataGenerator(
        name="RandomDataGen",
        phonon_displacement_maker=phonon_displacement_maker,
        n_struct=n_struct,
        uc=uc,
        cell_factor=cell_factor,
        std_dev=std_dev,
    ).make(structure=structure, mp_id=mp_id, supercell_matrix=supercell_matrix)
    jobs.append(random_datagen)

    flow = Flow(jobs, random_datagen.output)
    return Response(replace=flow)


@job
def get_iso_atom(structure_list: list[Structure]):
    """
    Job to collect all atomic species of the structures and starting VASP calculation of isolated atoms.

    Parameters
    ----------
    structure_list: list[Structure]
        list of pymatgen Structure objects
    """
    jobs = []
    all_species = list(
        {specie for s in structure_list for specie in s.types_of_species}
    )

    isoatoms = IsoAtomMaker().make(all_species=all_species)
    jobs.append(isoatoms)

    flow = Flow(
        jobs,
        {
            "species": all_species,
            "energies": isoatoms.output["energies"],
            "dirs": isoatoms.output["dirs"],
        },
    )
    return Response(replace=flow)
