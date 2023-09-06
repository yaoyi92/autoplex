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

    with open("results_sep_" + ".txt", 'w') as f:
        f.write("Pot Structure mpid RMS imagmodes(pot) imagmodes(dft) \n")

        comparison = CompareDFTMLMaker(name="comparetest")

        rms = comparison.rms_overall(quippyBS=mlphonon, vaspBS=dftphonon)

        rms2 = comparison.rms_overall_second_definition(quippyBS=mlphonon, vaspBS=dftphonon)

        comparison.rms_kdep_plot(whichkpath=2,
                                 filename=os.path.join(foldername, str(structure.composition.reduced_formula) + '_')
                                          + "_" + '_rms_phonons.eps')

        with open("results_sep_" + ".txt", 'a') as f:
            f.write("GAP" + ' ' + str(structure.composition.reduced_formula) + ' ' + str(rms) + '\n') + int(mpid)
            #TODO has img modes + ' ' + ' ' + str(ml.has_imag_modes(0.1)) + ' ' + str(dft.has_imag_modes(0.1)) +
