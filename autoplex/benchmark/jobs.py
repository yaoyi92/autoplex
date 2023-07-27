"""
Atomistic Jobs to Benchmark Potentials
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
# from PhoGap import PhonopyFiniteDisplacements, PhononsQuippyFiniteDisplacements, ComparePhononBS
from pathlib import Path

import itertools
import re
from jobflow import Flow, Response, job, Maker
# from pymatgen.core import Structure
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from atomate2.vasp.flows.phonons import PhononMaker
from atomate2.vasp.sets.core import StaticSetGenerator
from atomate2.vasp.jobs.base import BaseVaspMaker


@dataclass
class OptiQuippyMaker(Maker):
    '''
    Maker for GAP to phonon APEX calc
    '''

    name: str = "apex job maker"
    work_with_primitive: bool = False # True

    def make(self, structure: Structure, potential: Potential):
        '''
        Set up a quippy calculation
        :param structure:
        :param potential:
        :param smat:
        :return:
        '''

        self.initial_structure = structure
        self.potential = potential

        if self.work_with_primitive:
            structure = self._get_primitive_cell(self.initial_structure) #TODO doesn't work properly
        else:
            structure = self.initial_structure
        self.initial_structure_primitive = structure
        # print(self.initial_structure.lattice)
        self.optimized_structure = self._get_optimized_cell(structure, self.potential)

        return Response(output = self.optimized_structure)

    def read_yaml(self, filename):
        return load(filename)

    def _get_primitive_cell(self, structure) -> Structure:
        # returns a primitive cell
        """
        get a primitive structure
        Args:
            structure: Structure object

        Returns: Structure object

        """
        kpath = HighSymmKpath(structure, symprec = 0.0001)
        structure = kpath.prim
        return structure

    def _get_optimized_cell(self, structure, potential):
        atoms = self._get_ase_from_pmg(structure)
        atoms.set_calculator(potential)

        try:

            sf = UnitCellFilter(atoms)
            dyn = BFGS(sf)
            dyn.run(fmax = 1e-4)
            sf = StrainFilter(atoms)
            dyn = BFGS(sf)
            dyn.run(fmax = 1e-4)
            dyn = BFGS(atoms)
            dyn.run(fmax = 1e-4)
            sf = UnitCellFilter(atoms)
            dyn = BFGS(sf)
            dyn.run(fmax = 1e-4)

        except AttributeError:
            warnings.warn("No optimization is performed.")
        return self._get_pmg_from_ase(atoms)

    def _get_ase_from_pmg(self, structure):
        pymatgentoase = AseAtomsAdaptor()
        atoms_new = pymatgentoase.get_atoms(structure)
        return atoms_new

    def _get_pmg_from_ase(self, atoms):
        pymatgentoase = AseAtomsAdaptor()
        structure = pymatgentoase.get_structure(atoms)
        return structure


# static maker, parralization?

@dataclass
class StatQuippyMaker(Maker):
    """
    Class to carry out a single-point calculation for phonons using quippy and a ML fitted potential.
    Parameters
    ----------
    name
        Name of the job.
    """

    name: str = "phonon-static"

    def make(self, structure: Structure, potential: Potential):

        self.optimized_structure = structure
        self.potential = potential

        self.energy_optimized_structure = self._get_potential_energy(self.optimized_structure, self.potential)

        return Response(output = self.optimized_structure) #{"structure": self.optimized_structure, "energy": self.energy_optimized_structure,"phonon": self.phonon})

    def _get_ase_from_pmg(self, structure):
        pymatgentoase = AseAtomsAdaptor()
        atoms_new = pymatgentoase.get_atoms(structure)
        return atoms_new

    def _get_pmg_from_ase(self, atoms):
        pymatgentoase = AseAtomsAdaptor()
        structure = pymatgentoase.get_structure(atoms)
        return structure

    def _get_potential_energy(self, structure, potential):
        atoms = self._get_ase_from_pmg(structure)
        atoms.set_calculator(potential)
        return atoms.get_potential_energy()


@dataclass
class GenPhoBandDosQuippyMaker(Maker):
    """
    Class to generate phonon related stuff.
    Parameters
    ----------
    name
        Name of the job.
    """

    name: str = "phonon_band_dos"
    path_parameters: str = "phonopy.yaml"
    displacementdistance: float = 0.01
    set_phonons: bool = True
    set_thermal_conductivity: bool = False
    displacementdistancephonopy: float = 0.03
    max_distance_third_order = None
    temperature_range_kappa = range(50, 1001, 5)
    phonon: Phonopy = None

    if set_thermal_conductivity:
        # will do everything to calculate thermal conductivity
        kappa = _get_thermal_conductivity_matrix(temperatures = temperature_range_kappa)
        kappa_xx = []
        kappa_yy = []
        kappa_zz = []
        kappa_yz = []
        kappa_xz = []
        kappa_xy = []
        kappa_mean = []
        for value in kappa[0]:
            kappa_xx.append(value[0])
            kappa_yy.append(value[1])
            kappa_zz.append(value[2])
            kappa_yz.append(value[3])
            kappa_xz.append(value[4])
            kappa_xy.append(value[5])
            kappa_mean.append((value[0] + value[1] + value[2]) / 3.0)
    npoints_band: int = 101
    kpoint_density: float = 12000
    kpoint_density_phonopy: float = 1000

    def make(self, structure: Structure, potential: Potential, smat):

        self.optimized_structure = structure
        self.potential = potential
        self.smat = smat

        if self.set_phonons:
            if self.smat is None:
                self.smat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                for ilength, length in enumerate(self.optimized_structure.lattice.abc):
                    self.smat[ilength][ilength] = math.ceil(15.0 / length)

        self.phonon = self._get_phononobject_phonopy(self.optimized_structure, self.potential, smat = self.smat,
                                                     save_parameters = True, path = self.path_parameters,
                                                     displacement_distance = self.displacementdistance)
        #self.optimized_structure = get_pmg_structure(self.phonon.primitive)
        self.phonon_band_structure_pymatgen = self._get_bandstructure_calc(structure = self.optimized_structure,
                                                                           phonon = self.phonon,
                                                                           npoints_band = self.npoints_band)
        self.phonon_dos_pymatgen = self._get_dos_calc(structure = self.optimized_structure,
                                                      phonon = self.phonon,
                                                      kpoint_density = self.kpoint_density)
        return Response(
            output = {"band": self.phonon_band_structure_pymatgen, "dos": self.phonon_dos_pymatgen})  # output schema

    def _get_phononobject_phonopy(self, structure, potential, smat, save_parameters, path, displacement_distance=0.01):
        cell = get_phonopy_structure(structure)
        phonon = Phonopy(cell, smat, primitive_matrix = "auto", # [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                         factor = VaspToTHz)

        # displacements
        phonon.generate_displacements(distance = displacement_distance)
        print("[Phonopy] Atomic displacements:")
        disps = phonon.displacements
        for d in disps:
            print("[Phonopy] %d %s" % (d[0], d[1:]))

        supercells = phonon.supercells_with_displacements
        # Force calculations by calculator
        set_of_forces = []
        for scell in supercells:
            cell = Atoms(symbols = scell.get_chemical_symbols(),
                         scaled_positions = scell.get_scaled_positions(),
                         cell = scell.get_cell(),
                         pbc = True)
            cell.set_calculator(potential)
            # this part is adapted from: https://web.archive.org/web/20200610084959/https://github.com/phonopy/phonopy/blob/develop/example/ase/8Si-phonon.py
            # Copyright by Atsushi Togo
            forces = cell.get_forces()
            drift_force = forces.sum(axis = 0)
            print(("[Phonopy] Drift force:" + "%11.5f" * 3) % tuple(drift_force))
            for force in forces:
                force -= drift_force / forces.shape[0]
            set_of_forces.append(forces)

        phonon.produce_force_constants(forces = set_of_forces)
        if save_parameters:
            phonon.save(path)
        return phonon

    def kappa_at_temperature(self, temperature=100, whichvalue='xx'):
        for itemp, temp in enumerate(self.temperature_range_kappa):
            if temp == temperature:
                if whichvalue == "xx":
                    return self.kappa_xx[itemp]
                elif whichvalue == "yy":
                    return self.kappa_yy[itemp]
                elif whichvalue == "zz":
                    return self.kappa_zz[itemp]
                elif whichvalue == "yz":
                    return self.kappa_yz[itemp]
                elif whichvalue == "xz":
                    return self.kappa_xz[itemp]
                elif whichvalue == "xy":
                    return self.kappa_xy[itemp]
    def _get_bandstructure_calc(self, structure, phonon, npoints_band=51):
        # TODO add option to save yaml file with eigenvalues in future versions
        tempfilename = tempfile.gettempprefix() + '.yaml'
        kpath_dict, kpath_concrete = self.get_kpath(structure)
        qpoints, connections = get_band_qpoints_and_path_connections(kpath_concrete, npoints = npoints_band)
        phonon.run_band_structure(qpoints, path_connections = connections)
        phonon.write_yaml_band_structure(
            filename = tempfilename)
        bs_symm_line = get_ph_bs_symm_line(tempfilename, labels_dict = kpath_dict["kpoints"])
        os.remove(tempfilename)
        return bs_symm_line

    def _get_dos_calc(self, phonon, structure, kpoint_density):
        tempfilename = tempfile.gettempprefix() + '.yaml'
        kpoint = Kpoints.automatic_density(structure = structure, kppa = kpoint_density, force_gamma = True)
        phonon.run_mesh(kpoint.kpts[0])
        phonon.run_total_dos()
        phonon.write_total_dos(filename = tempfilename)
        dos = get_ph_dos(tempfilename)
        os.remove(tempfilename)
        return dos

    def has_imag_modes(self, tol=1e-5):
        return self.phonon_band_structure_pymatgen.has_imaginary_freq(tol = tol)

    def get_zero_point_energy(self):
        return self.phonon_dos_pymatgen.zero_point_energy(self.optimized_structure)

    def get_free_energy(self, temperature):
        return self.phonon_dos_pymatgen.helmholtz_free_energy(t = temperature, structure = self.optimized_structure)

    def get_kpath(self, structure: Structure):
        """
        get high-symmetry points in k-space
        Args:
            structure: Structure Object

        Returns:

        """
        kpath = KPathSeek(structure, symprec=0.0001) #HighSymmKpath(structure, symprec = 0.01)
        kpath_save = kpath.kpath
        labels = copy.deepcopy(kpath_save["path"])
        path = copy.deepcopy(kpath_save["path"])

        for ilabelset, labelset in enumerate(labels):
            for ilabel, label in enumerate(labelset):
                path[ilabelset][ilabel] = kpath_save["kpoints"][label]
        return kpath_save, path


@dataclass
class PlotPhoBandDosQuippyMaker(Maker):
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
class CompareVASPQuippyMaker(Maker):  # in pymatgen?
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

        print("band1: ", self.bands1, "\n band2: ", self.bands2)

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

        print("band1.1: ", self.bands1, "\n band2.1: ", self.bands2)

        # return Response(output = {"band1": self.bands1, "band2": self.bands2})

        band1 = np.sort(self.bands1, axis = 0)
        band2 = np.sort(self.bands2, axis = 0)

        print("band1.2: ", band1, "\n band2.2: ", band2)

        diff = band1 - band2
        return np.sqrt(np.mean(diff ** 2))

    # not used
    # def mean_absolute_error(self):
    #     band1 = np.sort(self.bands1, axis=0)
    #     band2 = np.sort(self.bands2, axis=0)
    #     diff_perc = (np.abs(band1 - band2)) / band1
    #     return np.mean(np.abs(diff_perc))

    #####

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

@job
def prepare_ML_for_phonons(
        pot: str,
        gapfile: str
):
    potential_filename = str(os.path.join(pot, gapfile))
    return Response(output = potential_filename)

@job
def ML_based_optimization(
        struc,
        potential_filename
):
    potential = Potential('IP GAP', param_filename = potential_filename)
    runner = OptiQuippyMaker(name = "testtesttest").make(structure = struc, potential = potential)

    return Response(output = runner.output)

@job
def ML_stat_calc(
        structure,
        potential_filename,
):
    potential = Potential('IP GAP', param_filename = potential_filename)
    stat = StatQuippyMaker(name = "teststat").make(structure = structure, potential = potential)
    return Response(output = stat.output)

@job
def ML_based_phonon_BS_DOS(
        statout,
        potential_filename,
        smat
):
    potential = Potential('IP GAP', param_filename = potential_filename)
    dosband = GenPhoBandDosQuippyMaker(name = "testdosband").make(
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
        plot = PlotPhoBandDosQuippyMaker(name = "testplot").make(dosband["band"], dosband["dos"],
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
            comparison = CompareVASPQuippyMaker(name = "comparetest")

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




