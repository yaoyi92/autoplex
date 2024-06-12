"""General AutoPLEX automation jobs."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jobflow import Flow, Response, job

if TYPE_CHECKING:
    from emmet.core.math import Matrix3D
    from pymatgen.core.structure import Structure

from autoplex.benchmark.phonons.flows import PhononBenchmarkMaker
from autoplex.data.phonons.flows import (
    DFTPhononMaker,
    IsoAtomMaker,
    MLPhononMaker,
    RandomStructuresDataGenerator,
)


@job
def complete_benchmark(  # this function was put here to prevent circular import
    ibenchmark_structure: int,
    benchmark_structure: Structure,
    min_length: float,
    ml_model: str,
    mp_ids,
    benchmark_mp_ids,
    add_dft_phonon_struct: bool,
    fit_input,
    symprec,
    phonon_displacement_maker,
    dft_references=None,
):
    """
    Need to add proper docstrings.

    Parameters
    ----------
    ibenchmark_structure
    benchmark_structure
    min_length
    ml_model
    mp_ids
    benchmark_mp_ids
    add_dft_phonon_struct
    fit_input
    symprec
    phonon_displacement_maker
    displacements
    dft_references

    """
    jobs = []
    collect_output = []
    for suffix in ["", "_wo_sigma", "_phonon", "_rand_struc"]:
        if Path(Path(ml_model) / f"gap_file{suffix}.xml").exists():
            add_data_ml_phonon = MLPhononMaker(
                min_length=min_length,
            ).make_from_ml_model(
                structure=benchmark_structure,
                ml_model=ml_model,
                suffix=suffix,
            )
            jobs.append(add_data_ml_phonon)
            if dft_references is None and benchmark_mp_ids is not None:
                if (
                    benchmark_mp_ids[ibenchmark_structure] in mp_ids
                ) and add_dft_phonon_struct:
                    dft_references = fit_input[benchmark_mp_ids[ibenchmark_structure]][
                        "phonon_data"
                    ]["001"]
                elif (
                    benchmark_mp_ids[ibenchmark_structure] not in mp_ids
                ) or (  # else?
                    add_dft_phonon_struct is False
                ):
                    dft_phonons = DFTPhononMaker(
                        symprec=symprec,
                        phonon_displacement_maker=phonon_displacement_maker,
                        born_maker=None,
                        min_length=min_length,
                    ).make(structure=benchmark_structure)
                    jobs.append(dft_phonons)
                    dft_references = dft_phonons.output

                add_data_bm = PhononBenchmarkMaker(name="Benchmark").make(
                    structure=benchmark_structure,
                    benchmark_mp_id=benchmark_mp_ids[ibenchmark_structure],
                    ml_phonon_task_doc=add_data_ml_phonon.output,
                    dft_phonon_task_doc=dft_references,
                )
            elif (
                dft_references is not None
                and not isinstance(dft_references, list)
                and benchmark_mp_ids is not None
            ):
                if benchmark_mp_ids[ibenchmark_structure] not in mp_ids:
                    add_data_bm = PhononBenchmarkMaker(name="Benchmark").make(
                        # this is important for re-using the same internally calculated DFT reference
                        # for looping through several settings
                        structure=benchmark_structure,
                        benchmark_mp_id=benchmark_mp_ids[ibenchmark_structure],
                        ml_phonon_task_doc=add_data_ml_phonon.output,
                        dft_phonon_task_doc=dft_references,
                    )
            else:
                add_data_bm = PhononBenchmarkMaker(name="Benchmark").make(
                    # this is important for using a provided DFT reference
                    structure=benchmark_structure,
                    benchmark_mp_id=benchmark_mp_ids[ibenchmark_structure],
                    ml_phonon_task_doc=add_data_ml_phonon.output,
                    dft_phonon_task_doc=dft_references[ibenchmark_structure],
                )
            jobs.append(add_data_bm)
            collect_output.append(add_data_bm.output)

    return Response(replace=jobs, output=collect_output)


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
    uc: bool = False,
    volume_custom_scale_factors: list[float] | None = None,
    volume_scale_factor_range: list[float] | None = None,
    rattle_std: float = 0.01,
    supercell_matrix: Matrix3D | None = None,
    distort_type: int = 0,
    n_structures: int = 10,
    min_distance: float = 1.5,
    angle_percentage_scale: float = 10,
    angle_max_attempts: int = 1000,
    rattle_type: int = 0,
    rattle_seed: int = 42,
    rattle_mc_n_iter: int = 10,
    w_angle: list[float] | None = None,
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
    uc: bool.
        If True, will generate randomly distorted structures (unitcells)
        and add static computation jobs to the flow.
    supercell_matrix: Matrix3D or None
        The matrix to construct the supercell.
    distort_type : int.
        0- volume distortion, 1- angle distortion, 2- volume and angle distortion. Default=0.
    distort_type : int.
        0- volume distortion, 1- angle distortion, 2- volume and angle distortion. Default=0.
    n_structures : int.
        Total number of distorted structures to be generated.
        Must be provided if distorting volume without specifying a range, or if distorting angles.
        Default=10.
    volume_scale_factor_range : list[float]
        [min, max] of volume scale factors.
        e.g. [0.90, 1.10] will distort volume +-10%.
    volume_custom_scale_factors : list[float]
        Specify explicit scale factors (if range is not specified).
        If None, will default to [0.90, 0.95, 0.98, 0.99, 1.01, 1.02, 1.05, 1.10].
    min_distance: float
        Minimum separation allowed between any two atoms.
        Default= 1.5A.
    angle_percentage_scale: float
        Angle scaling factor.
        Default= 10 will randomly distort angles by +-10% of original value.
    angle_max_attempts: int.
        Maximum number of attempts to distort structure before aborting.
        Default=1000.
    w_angle: list[float]
        List of angle indices to be changed i.e. 0=alpha, 1=beta, 2=gamma.
        Default= [0, 1, 2].
    rattle_type: int.
        0- standard rattling, 1- Monte-Carlo rattling. Default=0.
    rattle_std: float.
        Rattle amplitude (standard deviation in normal distribution).
        Default=0.01.
        Note that for MC rattling, displacements generated will roughly be
        rattle_mc_n_iter**0.5 * rattle_std for small values of n_iter.
    rattle_seed: int.
        Seed for setting up NumPy random state from which random numbers are generated.
        Default=42.
    rattle_mc_n_iter: int.
        Number of Monte Carlo iterations.
        Larger number of iterations will generate larger displacements.
        Default=10.
    """
    jobs = []

    random_datagen = RandomStructuresDataGenerator(
        name="RandomDataGen",
        phonon_displacement_maker=phonon_displacement_maker,
        n_structures=n_structures,
        uc=uc,
        rattle_std=rattle_std,
        distort_type=distort_type,
        min_distance=min_distance,
        rattle_seed=rattle_seed,
        rattle_type=rattle_type,
        angle_max_attempts=angle_max_attempts,
        angle_percentage_scale=angle_percentage_scale,
        rattle_mc_n_iter=rattle_mc_n_iter,
        w_angle=w_angle,
    ).make(
        structure=structure,
        mp_id=mp_id,
        supercell_matrix=supercell_matrix,
        volume_custom_scale_factors=volume_custom_scale_factors,
        volume_scale_factor_range=volume_scale_factor_range,
    )
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
