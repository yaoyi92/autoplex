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
    from pymatgen.core.structure import Structure

from autoplex.data.flows import RandomStruturesDataGenerator


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
def dft_phononpy_gen_data(
    structure: Structure, displacements, symprec, phonon_displacement_maker, min_length
):
    """
    Job to generate DFT reference database using phonopy to be used for fitting ML potentials.

    Parameters
    ----------
    name : str
        Name of the flow produced by this maker.
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
    flows = []
    dft_phonons_output = []
    dft_phonons_dir_output = []

    for displacement in displacements:
        dft_phonons = DFTPhononMaker(
            symprec=symprec,
            phonon_displacement_maker=phonon_displacement_maker,
            born_maker=None,
            displacement=displacement,
            min_length=min_length,
        ).make(structure=structure)
        dft_phonons = update_user_incar_settings(dft_phonons, {"NPAR": 4})
        flows.append(dft_phonons)
        dft_phonons_output.append(
            dft_phonons.output
        )  # CE: I have no better solution to this now
        dft_phonons_dir_output.append(dft_phonons.output.jobdirs.displacements_job_dirs)

    flow = Flow(flows, (dft_phonons_dir_output, dft_phonons_output))
    return Response(replace=flow)


@job
def dft_random_gen_data(
    structure: Structure, mp_id, phonon_displacement_maker, n_struct, sc
):
    """
    Job to generate random structured DFT reference database to be used for fitting ML potentials.

    Parameters
    ----------
    name : str
        Name of the flow produced by this maker.
    phonon_displacement_maker : .BaseVaspMaker or None
        Maker used to compute the forces for a supercell.
    mp_id:
        materials project id
    n_struct: int.
        The total number of randomly displaced structures to be generated.
    sc: bool.
        If True, will generate randomly distorted supercells structures
        and add static computation jobs to the flow
    """
    flows = []
    random_datagen = RandomStruturesDataGenerator(
        name="RandomDataGen",
        phonon_displacement_maker=phonon_displacement_maker,
        n_struct=n_struct,
        sc=sc,
    ).make(structure=structure, mp_id=mp_id)
    flows.append(random_datagen)

    flow = Flow(flows, random_datagen.output)
    return Response(replace=flow)
