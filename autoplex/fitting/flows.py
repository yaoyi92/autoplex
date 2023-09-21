"""
Flows consisting of jobs to fit ML potentials
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from jobflow import Flow, Maker
from autoplex.fitting.jobs import gapfit

__all__ = ["MLIPFitMaker"]


@dataclass
class MLIPFitMaker(Maker):
    """
    Maker to create ML potentials based on DFT data
    2. Step: Fit Potentials

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    For now GAP related parameters.

    """

    name: str = "MLpotentialFit"

    def make(
            self,
            species_list,
            iso_atom_energy,
            fitinput: list,
            ml_dir: str | Path | None = None,
            **fit_kwargs
    ):
        """
        Make flow to create ML potential fits.

        Parameters
        ----------
        for now GAP fit specific parameters
        """

        jobs = []
        GAPfit = gapfit(
            # mind the GAP # converting OUTCARs to a joint extended xyz file and running gap_fit with certain settings
            fitinput=fitinput,
            isolatedatoms=species_list,
            isolatedatomsenergy=iso_atom_energy,
            fit_kwargs=fit_kwargs,
        )
        jobs.append(GAPfit)

        # create a flow including all jobs
        flow = Flow(jobs, GAPfit.output)
        return flow
