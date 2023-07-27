"""
Jobs to fit ML potentials
"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import math
import tempfile
import warnings

import numpy as np
import matplotlib as mpl
from ase import Atoms
from ase.constraints import UnitCellFilter, StrainFilter
from ase.optimize import BFGS
from phonopy import Phonopy, load
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.units import VaspToTHz
from pymatgen.io.phonopy import get_ph_dos
from pymatgen.io.phonopy import get_phonopy_structure, get_ph_bs_symm_line, get_pmg_structure
from pymatgen.io.vasp import Kpoints
from pymatgen.phonon.plotter import PhononBSPlotter, PhononDosPlotter
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.symmetry.kpath import KPathSeek

from quippy.potential import Potential
from ase.io import read, write
from ase import Atom, Atoms
import logging
import subprocess
import os
from pathlib import Path

import itertools
import re
from jobflow import Flow, Response, job, Maker
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from atomate2.vasp.flows.phonons import PhononMaker
from atomate2.vasp.sets.core import StaticSetGenerator
from atomate2.vasp.jobs.base import BaseVaspMaker

@job
def gapfit(
        displacementinput,
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
):
    """
    job that prepares GAP fit input and fits the data using GAP. More ML methods (e.g. ACE) to follow.
    :param gap:
    :param default_sigma:
    :param energyparam:
    :param forcesparam:
    :param stressparam:
    :param sparse_jitter:
    :param do_copy_at:
    :param openmp:
    :param gpfile:
    :param isolatedatoms:
    :param isolatedatomsenergy:
    :param displacementinput:
    :param at_file:
    :param e0:
    :return: Job output
    """

    for displacements in displacementinput:
        for dis in displacements['dirs']:
            file = read(re.sub(r'^.*?/', '/', dis, count = 1) + "/OUTCAR.gz", index = ":")
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

        print("dir: ", directory)

    return Response(output = directory)
