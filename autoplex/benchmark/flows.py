"""
Flows consisting of jobs to benchmark ML potentials
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jobflow import Flow, Maker, OutputReference, job
from pymatgen.core.structure import Structure
from autoplex.benchmark.jobs import RMS


__all__ = ["PhononBenchmarkMaker"]


@dataclass
class PhononBenchmarkMaker(Maker):
    """
    Maker to benchmark ML potentials to DFT data
    3. Step: Evaluate Potentials

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.

    """

    name: str = "PhononBenchmark"

    def make(
            self,
            structure: Structure,
            mpid,  # list[MPID]
            ml_reference,
            dft_reference,
            **kwargs,
    ):
        """
        Make flow for benchmarking.

        Parameters
        ----------

        """

        jobs = []

        # number of points per path in phonon band structure #TODO einbauen
        kwargs.get("npoints_band", 51)
        kwargs.get("kpoint_density", 12000)


        rms = RMS(
            mlphonon=ml_reference,
            dftphonon=dft_reference,
            structure=structure
        )
        jobs.append(rms)

        # create a flow including all jobs
        flow = Flow(jobs, rms.output)
        return flow
