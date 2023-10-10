"""
Complete AutoPLEX -- Automated machine-learned Potential Landscape explorer -- jobs
"""
from __future__ import annotations
from pathlib import Path
from pymatgen.core.structure import Structure
from jobflow import Flow, job, Response
from atomate2.forcefields.jobs import GAPRelaxMaker, GAPStaticMaker
from atomate2.forcefields.flows.phonons import PhononMaker


@job
def PhononMLCalculationJob(
    ml_dir: str,
    structure: Structure,
    min_length: int = 20,
):
    jobs = []
    GAPPhonons = PhononMaker(
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
    jobs.append(GAPPhonons)

    flow = Flow(jobs, GAPPhonons.output)  # output for calculating RMS/benchmarking
    return Response(replace=flow)


@job
def CollectBenchmark(benchmark_structure: Structure, mpbm, rms, displacements):
    with open(
        f"results_{benchmark_structure.composition.get_reduced_formula_and_factor()[0]}.txt",
        "a",
    ) as file:
        file.write(
            f"Pot Structure mpid displacements RMS imagmodes(pot) imagmodes(dft) "
            f"\nGAP {benchmark_structure.composition.reduced_formula} {mpbm} {displacements} {rms} "
        )
        # TODO include which pot. method has been used (GAP, ACE, etc.)
        # TODO has img modes + ' ' + ' ' + str(ml.has_imag_modes(0.1)) + ' ' + str(dft.has_imag_modes(0.1))

    return Response
