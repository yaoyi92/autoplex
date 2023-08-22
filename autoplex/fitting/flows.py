"""
Flows consisting of jobs to fit ML potentials
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from jobflow import Flow, Maker, OutputReference, job
from pymatgen.core.structure import Structure
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
    # GAP specific input starts here
    at_file: str = 'at_file=trainGAP.xyz'
    e0: str = "e0={"
    order: int = 2
    cutoffgap: float = 6.5
    cutoffsoap: float = cutoffgap
    cutofftransgap: float = 0.5
    cutofftranssoap: float = 0.5
    n_sparse: int = 15
    covariance_type_gap: str = 'ard_se'
    deltagap: float = 2.00
    thetagap: float = 2.0
    sparsemethod: str = 'uniform'
    clusters: str = 'T'
    lmax: int = 6
    nmax: int = 12
    atomsigma: float = 0.5
    zeta: int = 4
    central_weight: float = 1.0
    n_sparse_soap: int = 7000
    deltasoap: float = 0.50
    f0: float = 0.0
    covariance_type_soap: str = 'dot_product'
    sparse_method: str = 'cur_points}'
    default_sigma_energy: float = 0.01
    default_sigma_force: float = 0.2
    default_sigma_virial: float = 0.2
    default_sigma_hessian: float = 0.0
    energy: str = 'energy'
    forces: str = 'forces'
    stress: str = 'virial'
    sparse_jitter: float = 1.0e-8
    do_copy_at: str = 'F'
    openmp_chunk_size: int = 10000
    gapfile: str = 'gap.xml'
    gap: str = 'gap={distance_Nb order=' + str(order) + ' cutoff=' + str(cutoffgap) + ' cutoff_transition_width=' + \
               str(cutofftransgap) + ' n_sparse=' + str(n_sparse) + ' covariance_type=' + covariance_type_gap + \
               ' delta=' + str(deltagap) + ' theta_uniform=' + str(thetagap) + ' sparse_method=' + sparsemethod + \
               ' compact_clusters=' + clusters + ':soap' + ' l_max=' + str(lmax) + ' n_max=' + str(nmax) + \
               ' atom_sigma=' + str(atomsigma) + ' zeta=' + str(zeta) + ' cutoff=' + str(cutoffsoap) + \
               ' cutoff_transition_width=' + str(cutofftranssoap) + ' central_weight=' + str(central_weight) + \
               ' n_sparse=' + str(n_sparse_soap) + ' delta=' + str(deltasoap) + ' f0=' + str(f0) + \
               ' covariance_type=' + covariance_type_soap + ' sparse_method=' + sparse_method
    default_sigma: str = ' default_sigma={' + str(default_sigma_energy) + ' ' + str(default_sigma_force) + ' ' + \
                         str(default_sigma_virial) + ' ' + str(default_sigma_hessian) + '}'
    energyparam: str = ' energy_parameter_name=' + energy
    forcesparam: str = ' force_parameter_name=' + forces
    stressparam: str = ' virial_parameter_name=' + stress
    jitter: str = ' sparse_jitter=' + str(sparse_jitter)
    copy: str = ' do_copy_at_file=' + do_copy_at
    openmp: str = ' openmp_chunk_size=' + str(openmp_chunk_size)
    gpfile: str = ' gp_file=' + gapfile

    def make(
            self,
            species_list,
            iso_atom_energy,
            fitinput: list,
            structurelist: list,
            ml_dir: str | Path | None = None
    ):
        """
        Make flow to create ML potential fits.

        Parameters
        ----------
        for now GAP fit specific parameters
        """

        jobs = []
        iso_atom_energy_list = []
        #for iso_atom in iso_atom_energy: iso_atom_energy_list.append(iso_atom.energy_per_atom)

        GAPfit = gapfit(
            # mind the GAP # converting OUTCARs to a joint extended xyz file and running gap_fit with certain settings
            fitinput=fitinput,
            isolatedatoms=species_list,
            isolatedatomsenergy=iso_atom_energy,
            at_file=self.at_file,
            e0=self.e0,
            gap=self.gap,
            default_sigma=self.default_sigma,
            energyparam=self.energyparam,
            forcesparam=self.forcesparam,
            stressparam=self.stressparam,
            sparse_jitter=self.jitter,
            do_copy_at=self.copy,
            openmp=self.openmp,
            gpfile=self.gpfile,
            gapfile=self.gapfile,
            structurelist=structurelist,
        )
        jobs.append(GAPfit)

        # create a flow including all jobs
        flow = Flow(jobs, GAPfit.output)
        return flow
