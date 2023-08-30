"""
Jobs to fit ML potentials
"""
from __future__ import annotations

import numpy as np
from ase.io import read, write
import subprocess
from pathlib import Path
import re
import os
from jobflow import Flow, Response, job

@job
def gapfit(
        fitinput: list,
        fitinputrand: list,
        isolatedatoms,
        isolatedatomsenergy,
        structurelist: list,
        kwargs
):
    """
    job that prepares GAP fit input and fits the data using GAP. More ML methods (e.g. ACE) to follow.

    """
    ### work in progress ###
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
    ### work in progress ###
    fit = []
    fit.extend(fitinput)
    fit.extend(fitinputrand)
    print("for debug: ", fit)
    for entry in fit:
        file = read(re.sub(r'^.*?/', '/', entry, count = 1) + "/OUTCAR.gz", index = ":")
        for i in file:  # credit goes to http://home.ustc.edu.cn/~lipai/scripts/ml_scripts/outcar2xyz.html
            xx, yy, zz, yz, xz, xy = -i.calc.results['stress'] * i.get_volume()
            i.info['virial'] = np.array([(xx, xy, xz), (xy, yy, yz), (xz, yz, zz)])
            del i.calc.results['stress']
            i.pbc = True
        write("trainGAP.xyz", file, append = True)



    for isoatom, isoenergy in zip(isolatedatoms, isolatedatomsenergy):
        if isoatom == isolatedatoms[-1]:
            e0 += str(isoatom) + ":" + str(isoenergy) + "}"
        else:
            e0 += str(isoatom) + ":" + str(isoenergy) + ":"

    with open('std_out.log', 'w') as f_std, open('std_err.log', 'w') as f_err:
        subprocess.call(['gap_fit', at_file, e0, gap, default_sigma, energyparam, forcesparam, stressparam,
                         sparse_jitter, do_copy_at, openmp, gpfile], stdout = f_std, stderr = f_err)

        directory = Path.cwd()

    return Response(output = {"dir": str(os.path.join(directory, gapfile)), "struclist": structurelist})
