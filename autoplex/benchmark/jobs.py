"""
Atomistic Jobs to Benchmark Potentials
"""
from __future__ import annotations

import os
from jobflow import Flow, Response, job, Maker
from autoplex.benchmark.utils import CompareDFTMLMaker
from pymatgen.core.structure import Structure


@job
def RMS(
        mlphonon,  # TODO include which pot. method has been used (GAP, ACE, etc.)
        dftphonon,
        structure: Structure,
        mpid
):
    foldername = "./"

    comparison = CompareDFTMLMaker(name="comparetest")

    rms = comparison.rms_overall(mlBS=mlphonon, dftBS=dftphonon)

    rms2 = comparison.rms_overall_second_definition(mlBS=mlphonon, dftBS=dftphonon)

    comparison.rms_kdep_plot(mlBS=mlphonon, dftBS=dftphonon, whichkpath=2,
                             filename=os.path.join(foldername, str(structure.composition.reduced_formula) + '_')
                                      + "_" + '_rms_phonons.eps')

    with open("results_" + ".txt", 'a') as f:
        f.write("Pot Structure mpid RMS RMS2 imagmodes(pot) imagmodes(dft) \nGAP" + ' ' +
                str(structure.composition.reduced_formula) + ' ' + str(rms) + str(rms2) + '\n') + int(mpid)
        #TODO has img modes + ' ' + ' ' + str(ml.has_imag_modes(0.1)) + ' ' + str(dft.has_imag_modes(0.1))
