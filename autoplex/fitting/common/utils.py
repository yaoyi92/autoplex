"""Utility functions for fitting jobs."""

from __future__ import annotations

import json
import os
import shutil
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

import ase
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.atoms import Atoms
from ase.constraints import voigt_6_to_full_3x3_stress
from ase.io import read, write
from atomate2.utils.path import strip_hostname
from scipy.spatial import ConvexHull
from sklearn.model_selection import StratifiedShuffleSplit


def load_gap_hyperparameter_defaults(gap_fit_parameter_file_path: str | Path):
    """
    Load gap fit default parameters from the json file.

    Parameters
    ----------
    gap_fit_parameter_file_path : str or Path.
        Path to gap-defaults.json.

    Returns
    -------
    dict
       gap fit default parameters.
    """
    with open(gap_fit_parameter_file_path, encoding="utf-8") as f:
        return json.load(f)


def gap_hyperparameter_constructor(
    gap_parameter_dict: dict,
    atoms_symbols: list | None = None,
    atoms_energies: list | None = None,
    include_two_body: bool = False,
    include_three_body: bool = False,
    include_soap: bool = False,
):
    """
    Construct a list of arguments needed to execute gap potential from the parameters' dict.

    Parameters
    ----------
    gap_parameter_dict : dict.
        dictionary with gap hyperparameters.
    atoms_symbols: list or None.
        List of atom symbols
    atoms_energies: list or None.
        List of isolated atoms energies
    include_two_body : bool.
        bool indicating whether to include two-body hyperparameters
    include_three_body : bool.
        bool indicating whether to include three-body hyperparameters
    include_soap : bool.
        bool indicating whether to include soap hyperparameters

    Returns
    -------
        list
           gap fit input parameter string.
    """
    # convert gap_parameter_dict to representation compatible with gap
    # if atoms_energies and atoms_symbols is not None:
    #     e0 = ":".join(
    #         [
    #             f"{iso_atom}:{iso_energy}"
    #             for iso_atom, iso_energy in zip(atoms_symbols, atoms_energies)
    #         ]
    #     )

    # Update the isolated atom energy argument
    # gap_parameter_dict["general"].update({"e0": e0})

    general = [f"{key}={value}" for key, value in gap_parameter_dict["general"].items()]

    two_body_params = " ".join(
        [
            f"{key}={value}"
            for key, value in gap_parameter_dict["twob"].items()
            if include_two_body is True
        ]
    )

    three_body_params = " ".join(
        [
            f"{key}={value}"
            for key, value in gap_parameter_dict["threeb"].items()
            if include_three_body is True
        ]
    )
    soap_params = " ".join(
        [
            f"{key}={value}"
            for key, value in gap_parameter_dict["soap"].items()
            if include_soap is True
        ]
    )
    # add separator between the arg types
    if include_two_body and include_three_body and include_soap:
        three_body_params = " :" + three_body_params
        soap_params = " :soap " + soap_params
    elif include_two_body and include_three_body and not include_soap:
        three_body_params = " :" + three_body_params
    elif (include_two_body or include_three_body) and include_soap:
        soap_params = " :soap " + soap_params
    elif include_soap and not include_three_body and not include_two_body:
        soap_params = "soap " + soap_params

    gap_hyperparameters = f"gap={{{two_body_params}{three_body_params}{soap_params}}}"

    return [*general, gap_hyperparameters]


def get_list_of_vasp_calc_dirs(flow_output):
    """
    Return a list of vasp_calc_dirs from PhononDFTMLDataGenerationFlow output.

    Parameters
    ----------
    flow_output: dict.
        PhononDFTMLDataGenerationFlow output

    Returns
    -------
    list.
        A list of vasp_calc_dirs
    """
    list_of_vasp_calc_dirs = []
    for output in flow_output.values():
        for output_type, dirs in output.items():
            if output_type != "phonon_data" and isinstance(dirs, list):
                list_of_vasp_calc_dirs.extend(*dirs)

    return list_of_vasp_calc_dirs


def outcar_2_extended_xyz(
    path_to_vasp_static_calcs: list,
    config_types: list[str] | None = None,
    xyz_file: str | None = None,
):
    """
    Parse all VASP OUTCARs and generates a trainGAP.xyz.

    Uses ase.io.read to parse the OUTCARs
    Adapted from https://lipai.github.io/scripts/ml_scripts/outcar2xyz.html

    Parameters
    ----------
    path_to_vasp_static_calcs : list.
        List of VASP static calculation directories.
    xyz_file: str or None
        a possibly already existing xyz file.
    config_types: list[str] or None
            list of config_types.
    """
    if config_types is None:
        config_types = ["bulk"] * len(path_to_vasp_static_calcs)

    for path, config_type in zip(path_to_vasp_static_calcs, config_types):
        # strip hostname if it exists in the path
        path_without_hostname = Path(strip_hostname(path)).joinpath("vasprun.xml.gz")
        # read the outcar
        file = read(path_without_hostname, index=":")
        for i in file:
            virial_list = -voigt_6_to_full_3x3_stress(i.get_stress()) * i.get_volume()
            i.info["REF_virial"] = " ".join(map(str, virial_list.flatten()))
            del i.calc.results["stress"]
            i.arrays["REF_forces"] = i.calc.results["forces"]
            del i.calc.results["forces"]
            i.info["REF_energy"] = i.calc.results["free_energy"]
            del i.calc.results["energy"]
            del i.calc.results["free_energy"]
            i.info["config_type"] = config_type
            i.pbc = True
        if xyz_file is not None:
            shutil.copy2(xyz_file, os.getcwd())
        write("vasp_ref.extxyz", file, append=True)


class Species:
    """Species class."""

    def __init__(self, atoms):
        self.atoms = atoms

    def get_species(self):
        """Get species."""
        sepcies_list = []

        for at in self.atoms:
            sym_all = at.get_chemical_symbols()
            syms = list(set(sym_all))
            for sym in syms:
                if sym in sepcies_list:
                    continue

                sepcies_list.append(sym)

        return sepcies_list

    def find_element_pairs(self, symb_list=None):
        """
        Find element pairs.

        Parameters
        ----------
        symb_list

        Returns
        -------
        pairs

        """
        species_list = self.get_species() if symb_list is None else symb_list

        pairs = []

        for i in range(len(species_list)):
            for j in range(i, len(species_list)):
                pair = (species_list[i], species_list[j])
                pairs.append(pair)

        return pairs

    def get_number_of_species(self):
        """Get number of species."""
        return int(len(self.get_species()))

    def get_species_Z(self):
        """Get species Z."""
        atom_numbers = []
        for atom_type in self.get_species():
            atom = Atoms(atom_type, [(0, 0, 0)])
            atom_numbers.append(int(atom.get_atomic_numbers()[0]))

        species_Z = "{"
        for i in range(len(atom_numbers) - 1):
            species_Z += str(atom_numbers[i]) + " "
        species_Z += str(atom_numbers[-1]) + "}"

        return species_Z


def flatten(o, recursive=False):
    """Flatten an iterable fully, but excluding Atoms objects."""
    iteration_list = []

    if recursive:
        for _ct, el in enumerate(o):
            if isinstance(el, Iterable) and not isinstance(
                el, (str, bytes, ase.atoms.Atoms, ase.Atoms)
            ):
                iteration_list += flatten(el, recursive=True)
            else:
                iteration_list += [el]
        return iteration_list

    return [item for sublist in o for item in sublist]


def gcm3_to_Vm(gcm3, mr, natoms=1):
    """Convert gcm3 to Vm."""
    return 1 / (natoms * (gcm3 / mr) * 6.022e23 / (1e8) ** 3)


def get_atomic_numbers(species):
    """Get atomic numbers."""
    atom_numbers = []
    for atom_type in species:
        atom = Atoms(atom_type, [(0, 0, 0)])
        atom_numbers.append(int(atom.get_atomic_numbers()[0]))

    return atom_numbers


def split_dataset(atoms, split_ratio):
    """Split the dataset."""
    atom_bulk = []
    atom_isolated_and_dimer = []
    for at in atoms:
        if (
            at.info["config_type"] != "dimer"
            and at.info["config_type"] != "isolated_atom"
        ):
            atom_bulk.append(at)
        else:
            atom_isolated_and_dimer.append(at)

    if len(atoms) != len(atom_bulk):
        atoms = atom_bulk

    average_energies = np.array([atom.info["REF_energy"] / len(atom) for atom in atoms])
    # sort by energy
    sorted_indices = np.argsort(average_energies)
    atoms = [atoms[i] for i in sorted_indices]
    average_energies = average_energies[sorted_indices]

    stratified_average_energies = pd.qcut(average_energies, q=2, labels=False)
    split = StratifiedShuffleSplit(n_splits=1, test_size=split_ratio, random_state=42)

    for train_index, test_index in split.split(atoms, stratified_average_energies):
        train_structures = [atoms[i] for i in train_index]
        test_structures = [atoms[i] for i in test_index]

    if atom_isolated_and_dimer:
        train_structures = atom_isolated_and_dimer + train_structures

    return train_structures, test_structures


def data_distillation(vasp_ref_dir, f_max):
    """For data distillation."""
    atoms = ase.io.read(vasp_ref_dir, index=":")

    atoms_distilled = []
    for at in atoms:
        forces = np.abs(at.arrays["REF_forces"])
        f_component_max = np.max(forces)

        if f_component_max < f_max:
            atoms_distilled.append(at)

    print(
        f"After distillation, there are still {len(atoms_distilled)} data points remaining."
    )

    return atoms_distilled


def rms_dict(x_ref, x_pred):
    """Take two datasets of the same shape and returns a dictionary containing RMS error data."""
    x_ref = np.array(x_ref)
    x_pred = np.array(x_pred)
    if np.shape(x_pred) != np.shape(x_ref):
        raise ValueError("WARNING: not matching shapes in rms")
    error_2 = (x_ref - x_pred) ** 2
    average = np.sqrt(np.average(error_2))
    std_ = np.sqrt(np.var(error_2))
    return {"rmse": average, "std": std_}


def energy_remain(in_file):
    """Plot the distribution of energy per atom on the output vs the input."""
    # read files
    in_atoms = ase.io.read(in_file, ":")
    ener_in = [
        at.info["REF_energy"] / len(at.get_chemical_symbols()) for at in in_atoms
    ]
    ener_out = [at.info["energy"] / len(at.get_chemical_symbols()) for at in in_atoms]
    _rms = rms_dict(ener_in, ener_out)
    return _rms["rmse"]


def extract_gap_label(xml_file_path):
    """Extract GAP label."""
    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    return root.tag


def plot_convex_hull(all_points, hull_points):
    """
    Plot convex hull.

    Parameters
    ----------
    all_points : ndarray.
        list of all points.
    hull_points: ndarray
        a possibly already existing xyz file.

    Returns
    -------
    None.

    """
    hull = ConvexHull(hull_points)

    plt.plot(all_points[:, 0], all_points[:, 1], "o", markersize=3, label="All Points")

    for i, simplex in enumerate(hull.simplices):
        if i == 0:
            plt.plot(
                hull_points[simplex, 0],
                hull_points[simplex, 1],
                "k-",
                label="Convex Hull",
            )
        else:
            plt.plot(hull_points[simplex, 0], hull_points[simplex, 1], "k-")

    plt.xlabel("Volume")
    plt.ylabel("Energy")
    plt.title("Convex Hull with All Points")
    plt.legend()
    plt.show()
