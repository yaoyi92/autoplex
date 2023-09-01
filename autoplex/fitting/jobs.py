"""
Jobs to fit ML potentials
"""
from __future__ import annotations

import json

import numpy as np
from ase.io import read, write
import subprocess
from pathlib import Path
import re
import os
from jobflow import Flow, Response, job
from dataclasses import dataclass, field

CurrentDir = Path(__file__).absolute().parent


@job
def gapfit(
        fitinput: list,
        fitinputrand: list,
        isolatedatoms,
        isolatedatomsenergy,
        gap_input=CurrentDir / "gap-defaults.json",
        twobody: bool = True,
        threebody: bool = False,
        soap: bool = True,
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

    with open(gap_input, "r") as infile:
        inputs = json.load(infile)

    e0: str = "{"

    for isoatom, isoenergy in zip(isolatedatoms, isolatedatomsenergy):
        if isoatom == isolatedatoms[-1]:
            e0 += str(isoatom) + ":" + str(isoenergy) + "}"
        else:
            e0 += str(isoatom) + ":" + str(isoenergy) + ":"
    # Updating the isolated atom energy
    inputs['general'].update({'e0': e0})
    # Overwriting the default gap_fit settings with user settings #TODO XPOT support
    for key in inputs:
        for key2 in fit_kwargs:
            if key == key2:
                inputs[key].update(fit_kwargs[key2])

    gap: str = GAPHyperparameterParser(inputs=inputs, twobody=twobody, threebody=threebody, soap=soap)
    general = [str(key) + "=" + str(inputs['general'][key]) for key in inputs['general']]

    with open('std_out.log', 'w') as f_std, open('std_err.log', 'w') as f_err:
        subprocess.call(['gap_fit'] + general + [gap], stdout=f_std, stderr=f_err)

        directory = Path.cwd()

    return Response(output=str(os.path.join(directory, inputs['general']['gp_file'])))


def GAPHyperparameterParser(inputs, twobody: bool = True, threebody: bool = False, soap: bool = True):
    twob: str = " ".join([f"{key}={value}" for key, value in inputs['twob'].items() if twobody == True])
    threeb: str = " ".join([f"{key}={value}" for key, value in inputs['threeb'].items() if threebody == True])
    SOAP: str = str(":soap " if soap == True else "") + " ".join(
        [f"{key}={value}" for key, value in inputs['soap'].items() if soap == True])
    gap: str = 'gap={'+ (twob + threeb + SOAP) + "}"

    return gap
