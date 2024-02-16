"""Flows consisting of jobs to fit ML potentials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
from jobflow import Flow, Maker

from autoplex.fitting.rss.flows import MLIPFitMaker as YbMLIPFitMaker
from autoplex.fitting.rss.flows import DataPreprocessing

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
        species_list: list,
        iso_atom_energy: list,
        fit_input: dict,
        ml_dir: str | Path | None = None,
        xyz_file: str | None = None,
        **fit_kwargs,
    ):
        """
        Make flow to create ML potential fits.

        Parameters
        ----------
        species_list : list.
            List of element names (str)
        iso_atom_energy : list.
            List of isolated atoms energy
        fit_input : dict.
            PhononDFTMLDataGenerationFlow output
        ml_dir: str or Path or None
            path to the ML potential file
        xyz_file: str or None
            a possibly already existing xyz file
        fit_kwargs : dict.
            dict including gap fit keyword args.
        """
        jobs = []
        # gap_fit_job = gapfit(
        #     fit_input=fit_input,
        #     isolated_atoms=species_list,
        #     isolated_atoms_energy=iso_atom_energy,
        #     xyz_file=xyz_file,
        #     fit_kwargs=fit_kwargs,
        # )
        data_prep_job = DataPreprocessing(
            split_ratio=0.4, regularization=True, distillation=True, f_max=40
        ).make(fit_input=fit_input, pre_database_dir=None, xyz_file=xyz_file)
        jobs.append(data_prep_job)
        gap_fit_job = YbMLIPFitMaker(mlip_type="GAP").make(
            database_dir=data_prep_job.output,
            isol_es=None,  # nequip={},
        )
        jobs.append(gap_fit_job)  # type: ignore

        # create a flow including all jobs
        return Flow(jobs, gap_fit_job.output)
