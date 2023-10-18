"""
Complete AutoPLEX -- Automated machine-learned Potential Landscape explorer -- jobs
"""
from __future__ import annotations
from pymatgen.core.structure import Structure
from jobflow import Flow, job, Response
from atomate2.forcefields.jobs import GAPRelaxMaker, GAPStaticMaker
from atomate2.forcefields.flows.phonons import PhononMaker


@job
def get_phonon_ml_calculation_jobs(
    ml_dir: str,
    structure: Structure,
    min_length: int = 20,
):
    """

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
