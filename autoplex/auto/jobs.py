"""General AutoPLEX automation jobs."""
from __future__ import annotations

from typing import TYPE_CHECKING

from atomate2.forcefields.flows.phonons import PhononMaker
from atomate2.forcefields.jobs import GAPRelaxMaker, GAPStaticMaker
from atomate2.vasp.flows.phonons import PhononMaker as DFTPhononMaker
from atomate2.vasp.powerups import (
    update_user_incar_settings,
)
from jobflow import Flow, Response, job

if TYPE_CHECKING:
    from emmet.core.math import Matrix3D
    from pymatgen.core.structure import Structure

from autoplex.data.flows import IsoAtomMaker, RandomStructuresDataGenerator

# This should be a maker rather than a job in a job
@job
def get_phonon_ml_calculation_jobs(
    ml_dir: str,
    structure: Structure,
    min_length: int = 20,
):
    """
    Get the PhononMaker job for ML-based phonon calculations.

    Parameters
    ----------
    ml_dir : str
        Path to gapfit.xml file
    structure: Structure
        pymatgen Structure object
    min_length: float
        min length of the supercell that will be built

    Returns
    -------
    A flow with GAP fit phonon jobs
    """
    jobs = []
    gap_phonons = PhononMaker(
        bulk_relax_maker=GAPRelaxMaker(
            potential_param_file_name=ml_dir,
            relax_cell=True,
            relax_kwargs={"interval": 500},
        ),
        phonon_displacement_maker=GAPStaticMaker(potential_param_file_name=ml_dir),
        static_energy_maker=GAPStaticMaker(potential_param_file_name=ml_dir),
        store_force_constants=False,
        generate_frequencies_eigenvectors_kwargs={"units": "THz"},
        min_length=min_length,
    ).make(structure=structure)
    jobs.append(gap_phonons)

    flow = Flow(jobs, gap_phonons.output)  # output for calculating RMS/benchmarking
    return Response(replace=flow)


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
    n_struct,
    uc,
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
        and add static computation jobs to the flow
    supercell_matrix: Matrix3D or None
        The matrix to construct the supercell.
    """
    jobs = []
    random_datagen = RandomStructuresDataGenerator(
        name="RandomDataGen",
        phonon_displacement_maker=phonon_displacement_maker,
        n_struct=n_struct,
        uc=uc,
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

    flow = Flow(jobs, {"species": all_species, "energies": isoatoms.output})
    return Response(replace=flow)
