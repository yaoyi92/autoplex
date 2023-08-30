"""
Flows consisting of jobs to benchmark ML potentials
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from jobflow import Flow, Maker, OutputReference, job
from pymatgen.core.structure import Structure


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
            structure_list: list[Structure],
            mpids: list,  # list[MPID]
            potDir: str | Path | None = None,
            **kwargs,
    ):
        """
        Make flow for benchmarking.

        Parameters
        ----------

        """

        jobs = []

        # prepare ML for phonon

        # number of points per path in phonon band structure #TODO einbauen
        kwargs.get("npoints_band", 51)
        kwargs.get("kpoint_density", 12000)

        start_from_files = False
        for ipot, pot in enumerate(potDir):
            # will be replace by forcefield GAP PhononMaker

            rms = RMS(
                distance = distance,
                struclist = structure_list,
                pot_nam = potential_names,
                foldername = ml_dir,
                dosband = RMSstep,
                dftphonon = PhononCollectOutput, # this later will be replaced by an independent DFT phonon run
                mpid = mpids
            )
            jobs.append(rms)

        # create a flow including all jobs
        flow = Flow(jobs)
        return flow
