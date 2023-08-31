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
from dataclasses import dataclass, field


@job
def gapfit(
        fitinput: list,
        fitinputrand: list,
        isolatedatoms,
        isolatedatomsenergy,
        fit_kwargs: dict = field(default_factory=dict),
):
    """
    job that prepares GAP fit input and fits the data using GAP. More ML methods (e.g. ACE) to follow.

    """
    fit = []
    fit.extend(fitinput)
    fit.extend(fitinputrand)
    for entry in fit:
        file = read(re.sub(r'^.*?/', '/', entry, count=1) + "/OUTCAR.gz", index=":")
        for i in file:  # credit goes to http://home.ustc.edu.cn/~lipai/scripts/ml_scripts/outcar2xyz.html
            xx, yy, zz, yz, xz, xy = -i.calc.results['stress'] * i.get_volume()
            i.info['virial'] = np.array([(xx, xy, xz), (xy, yy, yz), (xz, yz, zz)])
            del i.calc.results['stress']
            i.pbc = True
        write("trainGAP.xyz", file, append=True)

    for isoatom, isoenergy in zip(isolatedatoms, isolatedatomsenergy):
        if isoatom == isolatedatoms[-1]:
            e0 += str(isoatom) + ":" + str(isoenergy) + "}"
        else:
            e0 += str(isoatom) + ":" + str(isoenergy) + ":"

    #TODO reading hyperparamertes from default file
    #TODO let user overwrite/add hyperparameter settings

    with open('std_out.log', 'w') as f_std, open('std_err.log', 'w') as f_err: #TODO using hyperparameters from defualt json file
        subprocess.call(['gap_fit', at_file, e0, gap, default_sigma, energyparam, forcesparam, stressparam,
                         jitter, copy, openmp, gpfile], stdout=f_std, stderr=f_err)

        directory = Path.cwd()

    return Response(output=str(os.path.join(directory, gapfile)))

@job
def GAPHyperparameterParser():
    gap = "COMING SOON"
    # gap: str = 'gap={distance_Nb order=' + str(order) + ' cutoff=' + str(cutoff) + ' cutoff_transition_width=' + \
    #            str(cutofftrans) + ' n_sparse=' + str(n_sparse) + ' covariance_type=' + covariance_type + \
    #            ' delta=' + str(delta) + ' theta_uniform=' + str(theta) + ' sparse_method=' + sparsemethod + \
    #            ' compact_clusters=' + clusters + ':soap' + ' l_max=' + str(lmax) + ' n_max=' + str(nmax) + \
    #            ' atom_sigma=' + str(atomsigma) + ' zeta=' + str(zeta) + ' cutoff=' + str(cutoffsoap) + \
    #            ' cutoff_transition_width=' + str(cutofftranssoap) + ' central_weight=' + str(central_weight) + \
    #            ' n_sparse=' + str(n_sparse_soap) + ' delta=' + str(deltasoap) + ' f0=' + str(f0) + \
    #            ' covariance_type=' + covariance_type_soap + ' sparse_method=' + sparse_method
    # default_sigma: str = ' default_sigma={' + str(default_sigma_energy) + ' ' + str(default_sigma_force) + ' ' + \
    #                      str(default_sigma_virial) + ' ' + str(default_sigma_hessian) + '}'