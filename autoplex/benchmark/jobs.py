"""
Atomistic Jobs to Benchmark Potentials
"""
from __future__ import annotations

import os
import numpy as np
from pymatgen.phonon.plotter import PhononBSPlotter, PhononDosPlotter
from jobflow import Flow, Response, job, Maker
from pymatgen.core.structure import Structure


@job
def RMS(
        mlphonon,  # TODO include which pot. method has been used (GAP, ACE, etc.)
        dftphonon,
        structure: Structure,
        mpid
):
    def rms_overall():

        diff = np.array(mlBS['bands']) - np.array(dftBS['bands'])
        return np.sqrt(np.mean(diff ** 2))

    def rms_kdep():

        diff = np.array(mlBS['bands']) - np.array(dftBS['bands'])

        diff = np.transpose(diff)
        kpointdep = [np.sqrt(np.mean(diff[i] ** 2)) for i in range(len(diff))]
        return kpointdep

    def rms_kdep_plot(whichkpath=1, filename="rms.eps", format="eps"):

        rmskp = rms_kdep()

        if whichkpath == 1:
            plotter = PhononBSPlotter(bs=mlBS)
        elif whichkpath == 2:
            plotter = PhononBSPlotter(bs=dftBS)

        distances = []
        for element in plotter.bs_plot_data()["distances"]:
            distances.extend(element)
        import matplotlib.pyplot as plt
        plt.close("all")
        plt.plot(distances, rmskp)
        plt.xticks(ticks=plotter.bs_plot_data()["ticks"]["distance"],
                   labels=plotter.bs_plot_data()["ticks"]["label"])
        plt.xlabel("Wave vector")
        plt.ylabel("Phonons RMS (THz)")
        plt.savefig(filename, format=format)

    def compare_plot(filename="band_comparison.eps", img_format="eps"):
        plotter = PhononBSPlotter(bs=mlBS)
        plotter2 = PhononBSPlotter(bs=dftBS)
        new_plotter = plotter.plot_compare(plotter2)
        new_plotter.savefig(filename, format=img_format)
        new_plotter.close()

    def rms_overall_second_definition():
        # makes sure the frequencies are sorted by energy
        # otherwise the same as rms_overall

        band1 = np.sort(mlBS['bands'], axis=0)
        band2 = np.sort(dftBS['bands'], axis=0)

        diff = band1 - band2
        return np.sqrt(np.mean(diff ** 2))

    mlBS = mlphonon.phonon_bandstructure.as_dict()
    dftBS = dftphonon.phonon_bandstructure.as_dict()

    rms = rms_overall()

    rms2 = rms_overall_second_definition()

    rms_kdep_plot(whichkpath=2, filename=os.path.join(str(structure.composition.reduced_formula) + '_') + '_rms_phonons.eps')

    with open("results_" + ".txt", 'a') as f:
        f.write("Pot Structure mpid RMS RMS2 imagmodes(pot) imagmodes(dft) \nGAP" + ' ' + # TODO include which pot. method has been used (GAP, ACE, etc.)
                str(structure.composition.reduced_formula) + ' ' + str(rms) + str(rms2) + '\n') + int(mpid)
        #TODO has img modes + ' ' + ' ' + str(ml.has_imag_modes(0.1)) + ' ' + str(dft.has_imag_modes(0.1))



