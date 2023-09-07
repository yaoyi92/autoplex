"""
Functions, classes to benchmark ML potentials
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from pymatgen.phonon.plotter import PhononBSPlotter, PhononDosPlotter
from jobflow import Flow, Response, job, Maker


@dataclass
class CompareDFTMLMaker(Maker):
    """
    Class to compare DFT and ML calculations.
    Parameters
    ----------
    name
        Name of the job.
    """

    name: str = "compare_dft_ml"

    def make(self):
        return  # TODO

    def rms_overall(self, mlBS, dftBS):

        self.mlBS = mlBS
        self.dftBS = dftBS

        self.bands1 = self.mlBS.phonon_bandstructure.as_dict()['bands']
        self.bands2 = self.dftBS.phonon_bandstructure.as_dict()['bands']

        print("bands1", self.bands1, "bands2", self.bands2)

        diff = self.bands1 - self.bands2
        return np.sqrt(np.mean(diff ** 2))

    def rms_kdep(self):
        diff = self.bands1 - self.bands2

        diff = np.transpose(diff)
        kpointdep = [np.sqrt(np.mean(diff[i] ** 2)) for i in range(len(diff))]
        return kpointdep

    def rms_kdep_plot(self, whichkpath=1, filename="rms.eps", format="eps"):
        rms = self.rms_kdep()

        if whichkpath == 1:
            plotter = PhononBSPlotter(bs=self.mlBS)
        elif whichkpath == 2:
            plotter = PhononBSPlotter(bs=self.dftBS)

        distances = []
        for element in plotter.bs_plot_data()["distances"]:
            distances.extend(element)
        import matplotlib.pyplot as plt
        plt.close("all")
        plt.plot(distances, rms)
        plt.xticks(ticks=plotter.bs_plot_data()["ticks"]["distance"],
                   labels=plotter.bs_plot_data()["ticks"]["label"])
        plt.xlabel("Wave vector")
        plt.ylabel("Phonons RMS (THz)")
        plt.savefig(filename, format=format)

    def compare_plot(self, filename="band_comparison.eps", img_format="eps"):
        plotter = PhononBSPlotter(bs=self.mlBS)
        plotter2 = PhononBSPlotter(bs=self.dftBS)
        new_plotter = plotter.plot_compare(plotter2)
        new_plotter.savefig(filename, format=img_format)
        new_plotter.close()

    def rms_overall_second_definition(self, mlBS, dftBS):
        # makes sure the frequencies are sorted by energy
        # otherwise the same as rms_overall

        self.mlBS = mlBS
        self.dftBS = dftBS

        self.bands1 = self.mlBS.phonon_bandstructure.as_dict()['bands']
        self.bands2 = self.dftBS.phonon_bandstructure.as_dict()['bands']

        band1 = np.sort(self.bands1, axis=0)
        band2 = np.sort(self.bands2, axis=0)

        diff = band1 - band2
        return np.sqrt(np.mean(diff ** 2))
