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
        structure: Structure
):
    def rms_overall():

        diff = np.array(mlBS.as_dict()['bands']) - np.array(dftBS.as_dict()['bands'])
        return np.sqrt(np.mean(diff ** 2))

    def rms_kdep():

        diff = np.array(mlBS.as_dict()['bands']) - np.array(dftBS.as_dict()['bands'])

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

    rms_dis = []

    for dis_i, dis in enumerate(mlphonon):
        mlBS = dis.phonon_bandstructure
        dftBS = dftphonon[dis_i].phonon_bandstructure

        rms = rms_overall()
        rms_dis.append(rms)

        rms_kdep_plot(whichkpath=2,
                      filename=os.path.join(str(structure.composition.reduced_formula)) + '_rms_phonons.eps')

    return Response(output=rms_dis) #TODO TaskDoc




