"""
Flows consisting of jobs to benchmark ML potentials
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from jobflow import Flow, Maker, OutputReference, job
from pymatgen.core.structure import Structure
from autoplex.benchmark.jobs import prepare_ML_for_phonons, ML_based_optimization, ML_stat_calc, ML_based_phonon_BS_DOS


__all__ = ["BenchPress"]


@dataclass
class BenchPress(Maker):
    """
    Maker to create ML potentials based on DFT data
    3. Step: Evaluate Potentials

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.

    """

    name: str = "DFTbenchmark"

    def make(
            self,
            structure_list: list[Structure],
            mpids: list,  # list[MPID]
            potDir: str | Path | None = None,
    ):
        """
        Make flow for benchmarking..

        Parameters
        ----------

        """

        jobs = []

        # prepare ML for phonon

        # number of points per path in phonon band structure #TODO einbauen
        npoints_band = 51
        kpoint_density = 12000

        start_from_files = False
        for ipot, pot in enumerate(potDir):
            prepMLphon = prepare_ML_for_phonons(
                pot = pot,
                gapfile = self.gapfile,
            )
            jobs.append(prepMLphon)

            for i, struc in enumerate(structure_list):
                MLoptphon = ML_based_optimization(
                    struc = struc,
                    potential_filename = prepMLphon.output
                )
                jobs.append(MLoptphon)

                MLstat = ML_stat_calc(
                    structure = MLoptphon.output,
                    potential_filename = prepMLphon.output,
                )
                jobs.append(MLstat)

                MLBSDOS = ML_based_phonon_BS_DOS(
                    statout = MLstat.output,
                    potential_filename = prepMLphon.output,
                    smat = smat[i]
                )
                jobs.append(MLBSDOS)
                RMSstep.append(MLBSDOS.output)

                plotBSDOS = plot_BS_DOS(
                    distance = distance,
                    dosband = MLBSDOS.output,
                    struc = struc,
                    i = i,
                    pot_nam = potential_names[ipot]
                )
                jobs.append(plotBSDOS)

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
