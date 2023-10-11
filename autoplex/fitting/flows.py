"""
Flows consisting of jobs to fit ML potentials
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jobflow import Flow, Maker
from autoplex.fitting.jobs import gapfit

__all__ = ["MLIPFitMaker"]


@dataclass
class MLIPFitMaker(Maker):
    """
    Maker to fit ML potentials based on DFT data.

    Works only with gap at the moment.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    """

    name: str = "MLpotentialFit"

    def make(
        self,
        species_list,
        iso_atom_energy: list,
        fit_input: list,
        ml_dir: str | Path | None = None,
        **fit_kwargs,
    ):
        """
        Make flow to create ML potential fits.

        Parameters
        ----------
        species_list : list.
            List of species
        iso_atom_energy : list.
            List of isolated atoms energy
        fit_input : list.
            Mixed list of dictionary and lists
        ml_dir: str or Path or None
            path to the ML potential file
        fit_kwargs : dict.
            dict including gap fit keyword args.
        """
        jobs = []
        gap_fit_job = gapfit(
            # mind the GAP
            # converting OUTCARs to a joint extended xyz file
            # and running gap_fit with certain settings
            fit_input=fit_input,
            isolated_atoms=species_list,
            isolated_atoms_energy=iso_atom_energy,
            fit_kwargs=fit_kwargs,
        )
        jobs.append(gap_fit_job)  # type: ignore

        # create a flow including all jobs
        flow = Flow(jobs, gap_fit_job.output)
        return flow
