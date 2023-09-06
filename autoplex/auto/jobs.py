"""
Complete AutoPLEX -- Automated machine-learned Potential Landscape explorer -- jobs
"""

from pathlib import Path
from pymatgen.core.structure import Structure
from jobflow import Flow, job, Response
from atomate2.forcefields.jobs import GAPRelaxMaker, GAPStaticMaker
from atomate2.forcefields.flows.phonons import PhononMaker
from autoplex.benchmark.flows import PhononBenchmarkMaker

@job
def PhononDFTMLBenchmarkJob(
        structure_list: list[Structure],
        mpids: list,  # list[MPID]
        dft_reference,
        ml_dir: str | Path | None = None,
):
    jobs = []
    for struc_i, structure in enumerate(structure_list):
        GAPPhonons = PhononMaker(
            bulk_relax_maker=GAPRelaxMaker(potential_param_file_name=ml_dir, relax_cell=True,
                                           relax_kwargs={"interval": 500}),
            phonon_displacement_maker=GAPStaticMaker(potential_param_file_name=ml_dir),
            static_energy_maker=GAPStaticMaker(potential_param_file_name=ml_dir),
            store_force_constants=False,
            generate_frequencies_eigenvectors_kwargs={"units": "THz"}).make(
            structure=structure)
        jobs.append(GAPPhonons)
        benchmark = PhononBenchmarkMaker(name="Benchmark").make(structure=structure, mpid=mpids[struc_i], ml_reference=GAPPhonons.output, dft_reference=dft_reference)
        jobs.append(benchmark)

        flow = Flow(jobs) # TODO output # JaGeo: Probably good to define an output here to connect to the RMS values as well!
        return Response(replace=flow)