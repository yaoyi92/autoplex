"""
Jobs to create training data for ML potentials
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
def generate_random_displacement(
        displacements,
):
    random_displacements = []
    for dis in displacements:
        random = AseAtomsAdaptor.get_atoms(dis)
        random.rattle()
        random_displacements.append(AseAtomsAdaptor.get_structure(random))
    return random_displacements
