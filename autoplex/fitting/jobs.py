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
        fitinput,
        isolatedatoms,
        isolatedatomsenergy,
        at_file: str,
        e0: str,
        gap: str,
        default_sigma: str,
        energyparam: str,
        forcesparam: str,
        stressparam: str,
        sparse_jitter: str,
        do_copy_at: str,
        openmp: str,
        gpfile: str,
        gapfile: str,
        structurelist: list,
):
    """
    job that prepares GAP fit input and fits the data using GAP. More ML methods (e.g. ACE) to follow.

    """

    print("fit input: ", fitinput)

    for key in fitinput:
        for dir in fitinput[key]:
            file = read(re.sub(r'^.*?/', '/', dir, count = 1) + "/OUTCAR.gz", index = ":")
            for i in file:  # credit goes to http://home.ustc.edu.cn/~lipai/scripts/ml_scripts/outcar2xyz.html
                xx, yy, zz, yz, xz, xy = -i.calc.results['stress'] * i.get_volume()
                i.info['virial'] = np.array([(xx, xy, xz), (xy, yy, yz), (xz, yz, zz)])
                del i.calc.results['stress']
                i.pbc = True
                poten = i.get_potential_energy()
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
