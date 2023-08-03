"""
Atomistic Jobs to Benchmark Potentials
"""
from __future__ import annotations


from quippy.potential import Potential

import os
from jobflow import Flow, Response, job, Maker
from autoplex.benchmark.utils import OptimizationMLMaker, StatMLMaker, GenPhoBandDosMLMaker, PlotPhoBandDosMLMaker, CompareDFTMLMaker
from atomate2.forcefields.jobs import GAPRelaxMaker, GAPStaticMaker



@job
def prepare_ML_for_phonons(
        pot: str,
        gapfile: str
):
    potential_filename = str(os.path.join(pot, gapfile))
    return Response(output = potential_filename)

@job
def ML_based_optimization(
        structure,
       # potential_filename
):
    #potential = Potential('IP GAP', param_filename = potential_filename)
    optimize = GAPRelaxMaker(name = "test GAP").make(structure = structure) #OptimizationMLMaker(name = "testtesttest").make(structure = struc, potential = potential)
    return Response(output = optimize["final_structure"])

@job
def ML_stat_calc(
        structure,
        #potential_filename,
):
    print("is that a structure ? ", structure)
    #potential = Potential('IP GAP', param_filename = potential_filename)
    static = GAPStaticMaker(name = "test GAPstat").make(structure = structure) #StatMLMaker(name = "teststat").make(structure = structure, potential = potential)
    return Response(output = static["final_structure"])

@job
def ML_based_phonon_BS_DOS(
        statout,
        potential_filename,
        smat
):
    potential = Potential('IP GAP', param_filename = potential_filename)
    dosband = GenPhoBandDosMLMaker(name = "testdosband").make(
        structure = statout,
        potential = potential,
        smat = smat)

    return Response(output = dosband.output)

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




