"""Flows to benchmark ML potentials"""

from __future__ import annotations

from dataclasses import dataclass

from jobflow import Flow, Maker
from pymatgen.core.structure import Structure
from atomate2.common.schemas.phonons import PhononBSDOSDoc
from autoplex.benchmark.jobs import compute_bandstructure_benchmark_metrics


__all__ = ["PhononBenchmarkMaker"]


@dataclass
class PhononBenchmarkMaker(Maker):
    """
    Maker to benchmark ML potentials on reference DFT data

    Parameters
    ----------
    name : str
        Name of the flow produced by this maker.
    """

    name: str = "PhononBenchmark"

    def make(
        self,
        structure: Structure,
        mp_id: str,
        ml_phonon_task_doc: PhononBSDOSDoc,
        dft_phonon_task_doc: PhononBSDOSDoc,
        **kwargs,
    ):
        """
        Make flow for benchmarking.

        Parameters
        ----------
        structure :
            Pymatgen structures drawn from the Materials Project.
        mp_id: str.
            Materials project IDs for the structure
        ml_phonon_task_doc: PhononBSDOSDoc.
            ML potential generated pymatgen phonon band-structure object
        dft_phonon_task_doc: PhononBSDOSDoc.
            DFT generated pymatgen phonon band-structure object
        """
        jobs = []

        # number of points per path in phonon band structure einbauen
        kwargs.get("npoints_band", 51)
        kwargs.get("kpoint_density", 12000)

        rms = compute_bandstructure_benchmark_metrics(
            ml_phonon_task_doc=ml_phonon_task_doc,
            dft_phonon_task_doc=dft_phonon_task_doc,
            structure=structure,
        )
        jobs.append(rms)

        # create a flow including all jobs
        flow = Flow(jobs, rms.output)
        return flow
