"""Flows to benchmark ML potentials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from jobflow import Flow, Maker

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure
    from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine

from autoplex.benchmark.phonons.jobs import compute_bandstructure_benchmark_metrics

__all__ = ["PhononBenchmarkMaker"]


@dataclass
class PhononBenchmarkMaker(Maker):
    """
    Maker to benchmark ML potentials on reference DFT data.

    Parameters
    ----------
    name : str
        Name of the flow produced by this maker.
    """

    name: str = "PhononBenchmark"

    def make(
        self,
        structure: Structure,
        benchmark_mp_id: str,
        ml_phonon_bs: PhononBandStructureSymmLine,
        dft_phonon_bs: PhononBandStructureSymmLine,
        **kwargs,
    ):
        """
        Make flow for benchmarking.

        Parameters
        ----------
        structure :
            Pymatgen structures drawn from the Materials Project.
        benchmark_mp_id: str.
            Materials project IDs for the structure
        ml_phonon_bs: PhononBandStructureSymmLine.
            ML potential generated pymatgen phonon band-structure object
        dft_phonon_bs: PhononBandStructureSymmLine.
            DFT generated pymatgen phonon band-structure object
        """
        jobs = []

        # number of points per path in phonon band structure einbauen
        kwargs.get("npoints_band", 51)
        kwargs.get("kpoint_density", 12000)

        benchmark_job = compute_bandstructure_benchmark_metrics(
            ml_phonon_bs=ml_phonon_bs,
            dft_phonon_bs=dft_phonon_bs,
            structure=structure,
        )
        jobs.append(benchmark_job)

        # create a flow including all jobs
        return Flow(jobs, benchmark_job.output)
