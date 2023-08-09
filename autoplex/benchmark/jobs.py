"""
Atomistic Jobs to Benchmark Potentials
"""
from __future__ import annotations

import os
from jobflow import Flow, Response, job, Maker
from autoplex.benchmark.utils import PlotPhoBandDosMLMaker, CompareDFTMLMaker


@job
def plot_BS_DOS(
        distance,
        dosband,
        struc,
        i,
        pot_nam
    ):
    for displ_dist in distance:
        plot = PlotPhoBandDosMLMaker(name = "testplot").make(dosband["band"], dosband["dos"],
                                                                     bandname = os.path.join(
                                                                         str(struc.composition.reduced_formula) + "_test_"
                                                                         + str(i) + "_" + str(pot_nam) + "_" + str(
                                                                             displ_dist) + "_gap_band_structure.eps"),
                                                                     ylim = [-1, 16], dosname = os.path.join(
                    str(struc.composition.reduced_formula) + "_test_" + str(i) + "_" + str(pot_nam) + "_" +
                    str(displ_dist) + "_gap_dos_structure.eps"))

@job
def RMS(
        distance,
        struclist,
        pot_nam,
        foldername,
        dosband,
        dftphonon,
        mpid
):
    for displ_dist in distance:
        for istruc, struc in enumerate(struclist):
            phonon_band_vasp = dftphonon[istruc]
            phonon_ml = dosband[istruc]
            with open("results_sep_" + str(displ_dist) + ".txt", 'w') as f:
                f.write("Pot Structure mpid RMS imagmodes(pot) imagmodes(dft) \n")

            # try:
            comparison = CompareDFTMLMaker(name = "comparetest")

            rms = comparison.rms_overall(quippyBS = phonon_ml["band"], vaspBS = phonon_band_vasp)

            rms2 = comparison.rms_overall_second_definition(quippyBS = phonon_ml["band"],
                                                            vaspBS = phonon_band_vasp)

            comparison.rms_kdep_plot(whichkpath = 2,
                                     filename = os.path.join(foldername,
                                                             str(struc.composition.reduced_formula) + '_' + str(
                                                                 pot_nam[istruc]) + "_" + str(
                                                                 displ_dist) + '_''_rms_phonons.eps'))

            with open("results_sep_" + str(displ_dist) + ".txt", 'a') as f:
                f.write(str(pot_nam[istruc]) + ' ' + str(struc.composition.reduced_formula) + ' ' + str(rms) + '\n') + int(mpid[istruc])
                # + ' ' + ' ' + str( runner.has_imag_modes(0.1)) + ' ' + str(runner_castep.has_imag_modes(0.1)) +




