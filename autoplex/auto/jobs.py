"""
Complete AutoPLEX -- Automated machine-learned Potential Landscape explorer -- jobs
"""

from pathlib import Path
from pymatgen.core.structure import Structure
from jobflow import Flow, job, Response
from atomate2.forcefields.jobs import GAPRelaxMaker, GAPStaticMaker
from atomate2.forcefields.flows.phonons import PhononMaker


@job
def PhononMLCalculationJob(
        structure: Structure,
        displacements,
        min_length: int = 20,
        ml_dir: str | Path | None = None,
):
    jobs = []
    GAPphonons_output = []
    for displacement in displacements:
        GAPPhonons = PhononMaker(
            bulk_relax_maker=GAPRelaxMaker(potential_param_file_name=ml_dir, relax_cell=True,
                                           relax_kwargs={"interval": 500}),
            phonon_displacement_maker=GAPStaticMaker(potential_param_file_name=ml_dir),
            static_energy_maker=GAPStaticMaker(potential_param_file_name=ml_dir),
            store_force_constants=False,
            generate_frequencies_eigenvectors_kwargs={"units": "THz"}, displacement=displacement,
            min_length=min_length).make(structure=structure)
        jobs.append(GAPPhonons)
        GAPphonons_output.append(GAPPhonons.output)

    flow = Flow(jobs, GAPphonons_output)  # output for calculating RMS/benchmarking
    return Response(replace=flow)


@job
def CollectBenchmark(
        structure_list: list[Structure],
        mpids,
        rms_dis,
        displacements
):
    with open("results_" + ".txt", 'a') as f:
        f.write("Pot Structure mpid displacement RMS imagmodes(pot) imagmodes(dft) \nGAP ")
        # TODO include which pot. method has been used (GAP, ACE, etc.)

    for dis, rms in enumerate(rms_dis):
        for struc_i, structure in enumerate(structure_list):
            with open("results_" + ".txt", 'a') as f:
                f.write(f"{structure.composition.reduced_formula} {mpids[struc_i]} {displacements[dis]} {rms[struc_i]} \nGAP ")
                # TODO has img modes + ' ' + ' ' + str(ml.has_imag_modes(0.1)) + ' ' + str(dft.has_imag_modes(0.1))

    return Response
