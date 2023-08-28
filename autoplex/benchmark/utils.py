"""
Functions, classes to benchmark ML potentials
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from pymatgen.phonon.plotter import PhononBSPlotter, PhononDosPlotter
from jobflow import Flow, Response, job, Maker



@dataclass
class PlotPhoBandDosMLMaker(Maker):
    """
    Class to plot phonon related stuff.
    Parameters
    ----------
    name
        Name of the job.
    """

    name: str = "plot_phonon_band_dos"

    def make(self, phonon_band_structure_pymatgen, phonon_dos_pymatgen, bandname: str, ylim, dosname: str):
        self.phonon_band_structure_pymatgen = phonon_band_structure_pymatgen
        self.phonon_dos_pymatgen = phonon_dos_pymatgen
        self.bandname = bandname
        self.ylim = ylim
        self.dosname = dosname

        self.save_plot_band_structure(filename = self.bandname, ylim = self.ylim)
        self.save_plot_dos(filename = self.dosname)
        return Response

    def save_plot_band_structure(self, filename, img_format="eps", units="thz", ylim=None):
        plotter = PhononBSPlotter(bs = self.phonon_band_structure_pymatgen)
        plotter.save_plot(filename, img_format = img_format, units = units, ylim = ylim)

    def save_plot_dos(self, filename, img_format="eps", units="thz", label="total dos"):
        plotter2 = PhononDosPlotter(stack = False, sigma = None)
        plotter2.add_dos(label = label, dos = self.phonon_dos_pymatgen)
        plotter2.save_plot(filename = filename, img_format = img_format, units = units)

    def save_kappa_plot(self, mean_exp_data_T=None, mean_exp_data=None, exp_data_xx=None, exp_data_yy=None,
                        exp_data_zz=None, mean_xx_yy_zz=True, xx=True, yy=True, zz=True, yz=True,
                        xz=True, xy=True, filename="kappa.eps", format='eps'):

        import matplotlib.pyplot as plt
        plt.close("all")

        if xx:
            plt.plot(list(self.temperature_range_kappa), self.kappa_xx, label = "xx")
        if yy:
            plt.plot(list(self.temperature_range_kappa), self.kappa_yy, label = "yy")
        if zz:
            plt.plot(list(self.temperature_range_kappa), self.kappa_zz, label = "zz")
        if yz:
            plt.plot(list(self.temperature_range_kappa), self.kappa_yz, label = "yz")
        if xz:
            plt.plot(list(self.temperature_range_kappa), self.kappa_xz, label = "xz")
        if xy:
            plt.plot(list(self.temperature_range_kappa), self.kappa_xy, label = "xy")
        if mean_xx_yy_zz:
            plt.plot(list(self.temperature_range_kappa), self.kappa_mean, label = "mean_xx_yy_zz")
        if mean_exp_data_T is not None and mean_exp_data is not None:
            plt.plot(mean_exp_data_T, mean_exp_data, label = "benchmark")
        if mean_exp_data_T is not None and exp_data_xx is not None:
            plt.plot(mean_exp_data_T, exp_data_xx, label = "benchmark_xx")
        if mean_exp_data_T is not None and exp_data_yy is not None:
            plt.plot(mean_exp_data_T, exp_data_yy, label = "benchmark_yy")
        if mean_exp_data_T is not None and exp_data_zz is not None:
            plt.plot(mean_exp_data_T, exp_data_zz, label = "benchmark_zz")
        plt.xlabel("Temperature (K)")
        plt.ylabel("Kappa (W/(mK))")
        plt.legend()
        plt.savefig(filename, format = format)


@dataclass
class CompareDFTMLMaker(Maker):  # in pymatgen?
    """
    Class to compare VASP and quippy calculations.
    Parameters
    ----------
    name
        Name of the job.
    """

    name: str = "compare_vasp_quippy"

    # def __init__(self, phoncalc1, phoncalc2, bs1, bs2):
    #     super().__init__()
    #     self.phoncalc1 = phoncalc1
    #     self.phoncalc2 = phoncalc2
    #
    #     self.bs1 = bs1
    #     self.bs2 = bs2
    #     self.bands1 = self.bs1.bands
    #     self.bands2 = self.bs2.bands

    def make(self):
        return  # idk

    def rms_overall(self, quippyBS, vaspBS):

        self.quippyBS = quippyBS
        self.vaspBS = vaspBS

        self.bands1 = self.quippyBS.bands
        self.bands2 = self.vaspBS.as_dict()['bands']

        # return Response(output = {"band1": self.bands1, "band2": self.bands2})
        diff = self.bands1 - self.bands2
        return np.sqrt(np.mean(diff ** 2))

    def compare_zpe(self):
        return

    def get_rms_zpe(self):
        zpe1 = self.phoncalc1.get_zero_point_energy()
        zpe2 = self.phoncalc2.get_zero_point_energy()
        return np.sqrt((zpe1 - zpe2) ** 2)

    def get_rms_free_energy(self, startt=0, stopt=1000, stept=10):
        list1 = []
        list2 = []
        for t in range(startt, stopt + stept, stept):
            list1.append(self.phoncalc1.get_free_energy(temperature = t))
            list2.append(self.phoncalc2.get_free_energy(temperature = t))
        diff = np.array(list1) - np.array(list2)
        return np.sqrt(np.mean(diff ** 2))

    @property
    def compare_phonon_BS(self):
        return

    def rms_kdep(self):
        diff = self.bands1 - self.bands2

        diff = np.transpose(diff)
        kpointdep = [np.sqrt(np.mean(diff[i] ** 2)) for i in range(len(diff))]
        return kpointdep

    def rms_kdep_plot(self, whichkpath=1, filename="rms.eps", format="eps"):
        rms = self.rms_kdep()

        if whichkpath == 1:
            plotter = PhononBSPlotter(bs = self.quippyBS)
        elif whichkpath == 2:
            plotter = PhononBSPlotter(bs = self.vaspBS)

        distances = []
        for element in plotter.bs_plot_data()["distances"]:
            distances.extend(element)
        import matplotlib.pyplot as plt
        plt.close("all")
        plt.plot(distances, rms)
        plt.xticks(ticks = plotter.bs_plot_data()["ticks"]["distance"],
                   labels = plotter.bs_plot_data()["ticks"]["label"])
        plt.xlabel("Wave vector")
        plt.ylabel("Phonons RMS (THz)")
        plt.savefig(filename, format = format)

    def compare_plot(self, filename="band_comparison.eps", img_format="eps"):
        plotter = PhononBSPlotter(bs = self.quippyBS)
        plotter2 = PhononBSPlotter(bs = self.vaspBS)
        new_plotter = plotter.plot_compare(plotter2)
        new_plotter.savefig(filename, format = img_format)
        new_plotter.close()

    def rms_overall_second_definition(self, quippyBS, vaspBS):
        # makes sure the frequencies are sorted by energy
        # otherwise the same as rms_overall

        self.quippyBS = quippyBS
        self.vaspBS = vaspBS

        self.bands1 = self.quippyBS.bands
        self.bands2 = self.vaspBS.as_dict()['bands']

        band1 = np.sort(self.bands1, axis = 0)
        band2 = np.sort(self.bands2, axis = 0)

        diff = band1 - band2
        return np.sqrt(np.mean(diff ** 2))

    def calculate_rms(self, exp_data_x=None, exp_data_y=None):
        to_compare = []
        exp_data_y2 = []
        for ix, x in enumerate(exp_data_x):
            for itemp, temp in enumerate(self.temperature_range_kappa):
                if x == temp:
                    to_compare.append(self.kappa_mean[itemp])
                    exp_data_y2.append(exp_data_y[ix])
        diff = np.array(to_compare) - np.array(exp_data_y2)
        return np.sqrt(np.mean(diff ** 2))

    def calculate_rms_xyz(self, exp_data_T=None, exp_data_xx=None, exp_data_yy=None, exp_data_zz=None):
        to_compare = []
        exp_data_y2 = []
        for number in range(0, 3):
            for ix, x in enumerate(exp_data_T):
                for itemp, temp in enumerate(self.temperature_range_kappa):
                    if x == temp:
                        if number == 0:
                            to_compare.append(self.kappa_xx[itemp])
                            exp_data_y2.append(exp_data_xx[ix])
                        elif number == 1:
                            to_compare.append(self.kappa_yy[itemp])
                            exp_data_y2.append(exp_data_yy[ix])
                        elif number == 2:
                            to_compare.append(self.kappa_zz[itemp])
                            exp_data_y2.append(exp_data_zz[ix])

        diff = np.array(to_compare) - np.array(exp_data_y2)
        return np.sqrt(np.mean(diff ** 2))
