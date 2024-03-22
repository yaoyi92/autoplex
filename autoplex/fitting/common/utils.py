"""Utility functions for fitting jobs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from itertools import combinations
from pathlib import Path

import ase
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.atoms import Atoms
from ase.constraints import voigt_6_to_full_3x3_stress
from ase.io import read, write
from ase.neighborlist import NeighborList, natural_cutoffs
from atomate2.utils.path import strip_hostname
from scipy.spatial import ConvexHull
from sklearn.model_selection import StratifiedShuffleSplit

current_dir = Path(__file__).absolute().parent
GAP_DEFAULTS_FILE_PATH = current_dir / "gap-defaults.json"


def gap_fitting(
    db_dir: str | Path,
    include_two_body: bool = True,
    include_three_body: bool = False,
    include_soap: bool = True,
    path_to_default_hyperparameters: Path | str = GAP_DEFAULTS_FILE_PATH,
    num_processes: int = 32,
    auto_delta: bool = True,
    glue_xml: bool = False,
    fit_kwargs: dict | None = None,  # pylint: disable=E3701
):
    """
    GAP fit and validation job.

    Parameters
    ----------
    db_dir: str or path.
        Path to database directory.
    path_to_default_hyperparameters : str or Path.
        Path to gap-defaults.json.
    include_two_body : bool.
        bool indicating whether to include two-body hyperparameters
    include_three_body : bool.
        bool indicating whether to include three-body hyperparameters
    include_soap : bool.
        bool indicating whether to include soap hyperparameters
    num_processes: int.
        Number of processes used for gap_fit
    auto_delta: bool
        automatically determine delta for 2b, 3b and soap terms.
    glue_xml: bool
        use the glue.xml core potential instead of fitting 2b terms.
    fit_kwargs: dict.
        optional dictionary with parameters for gap fitting with keys same as
        gap-defaults.json.

    Returns
    -------
    dict[str, float]
        A dictionary with train_error, test_error

    """
    mlip_path: Path = prepare_fit_environment(db_dir, Path.cwd(), glue_xml)

    db_atoms = ase.io.read(os.path.join(db_dir, "train.extxyz"), index=":")
    train_data_path = os.path.join(db_dir, "train_with_sigma.extxyz")
    test_data_path = os.path.join(db_dir, "test.extxyz")

    gap_default_hyperparameters = load_gap_hyperparameter_defaults(
        gap_fit_parameter_file_path=path_to_default_hyperparameters
    )

    for parameter in gap_default_hyperparameters:
        if fit_kwargs:
            for arg in fit_kwargs:
                if parameter == arg:
                    gap_default_hyperparameters[parameter].update(fit_kwargs[arg])
                if glue_xml:
                    for item in fit_kwargs["general"].items():
                        if item == ("core_param_file", "glue.xml"):
                            gap_default_hyperparameters["general"].update(
                                {"core_param_file": "glue.xml"}
                            )
                            gap_default_hyperparameters["general"].update(
                                {"core_ip_args": "{IP Glue}"}
                            )

    if include_two_body:
        gap_default_hyperparameters["general"].update({"at_file": train_data_path})
        if auto_delta:
            delta_2b = calculate_delta(db_atoms, "REF_energy")
            gap_default_hyperparameters["twob"].update({"delta": delta_2b})

        fit_parameters_list = gap_hyperparameter_constructor(
            gap_parameter_dict=gap_default_hyperparameters,
            include_two_body=include_two_body,
        )

        run_gap(num_processes, fit_parameters_list)
        run_quip(num_processes, train_data_path, "gap_file.xml", "quip_train.extxyz")

    if include_three_body:
        gap_default_hyperparameters["general"].update({"at_file": train_data_path})
        if auto_delta:
            delta_3b = energy_remain("quip_train.extxyz")
            gap_default_hyperparameters["threeb"].update({"delta": delta_3b})

        fit_parameters_list = gap_hyperparameter_constructor(
            gap_parameter_dict=gap_default_hyperparameters,
            include_two_body=include_two_body,
            include_three_body=include_three_body,
        )

        run_gap(num_processes, fit_parameters_list)
        run_quip(num_processes, train_data_path, "gap_file.xml", "quip_train.extxyz")

    if include_soap:
        delta_soap = (
            energy_remain("quip_train.extxyz")
            if include_two_body or include_three_body
            else 1
        )
        gap_default_hyperparameters["general"].update({"at_file": train_data_path})
        if auto_delta:
            gap_default_hyperparameters["soap"].update({"delta": delta_soap})

        fit_parameters_list = gap_hyperparameter_constructor(
            gap_parameter_dict=gap_default_hyperparameters,
            include_two_body=include_two_body,
            include_three_body=include_three_body,
            include_soap=include_soap,
        )

        run_gap(num_processes, fit_parameters_list)
        run_quip(num_processes, train_data_path, "gap_file.xml", "quip_train.extxyz")

    # Calculate training error
    train_error = energy_remain("quip_train.extxyz")
    print("Training error of MLIP (eV/at.):", round(train_error, 4))

    # Calculate testing error
    run_quip(num_processes, test_data_path, "gap_file.xml", "quip_test.extxyz")
    test_error = energy_remain("quip_test.extxyz")
    print("Testing error of MLIP (eV/at.):", round(test_error, 4))

    return {
        "train_error": train_error,
        "test_error": test_error,
        "mlip_path": mlip_path,
    }


def check_convergence(test_error):
    """
    Check the convergence of the fit.

    Parameters
    ----------
    test_error:
        The error of the test data.

    Returns
    -------
    The convergence bool.
    """
    convergence = False
    if test_error < 0.01:
        convergence = True

    return convergence


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
                if output_type == "rand_struc_dir":
                    flat_dirs = [[item for sublist in dirs for item in sublist]]
                    list_of_vasp_calc_dirs.extend(*flat_dirs)
                else:
                    list_of_vasp_calc_dirs.extend(*dirs)

    return list_of_vasp_calc_dirs


def vaspoutput_2_extended_xyz(
    path_to_vasp_static_calcs: list,
    config_types: list[str] | None = None,
    data_types: list[str] | None = None,
    regularization: float = 0.1,
    f_min: float = 0.01,  # unit: eV Å-1
    atom_wise_regularization: bool = True,
):
    """
    Parse all VASP output files (vasprun.xml/OUTCAR) and generates a vasp_ref.extxyz.

    Uses ase.io.read to parse the OUTCARs
    Adapted from https://lipai.github.io/scripts/ml_scripts/outcar2xyz.html

    Parameters
    ----------
    path_to_vasp_static_calcs : list.
        List of VASP static calculation directories.
    config_types: list[str] or None
            list of config_types.
    data_types: list[str] or None
            track the data type (phonon or random).
    regularization: float
        regularization value for the atom-wise force components.
    f_min: float
        minimal force cutoff value for atom-wise regularization.
    atom_wise_regularization: bool
        for including atom-wise regularization.
    """
    if config_types is None:
        config_types = ["bulk"] * len(path_to_vasp_static_calcs)
    if data_types is None:
        data_types = ["other"] * len(path_to_vasp_static_calcs)

    for path, config_type, data_type in zip(
        path_to_vasp_static_calcs, config_types, data_types
    ):
        # strip hostname if it exists in the path
        path_without_hostname = Path(strip_hostname(path)).joinpath("vasprun.xml.gz")
        # read the outcar
        file = read(path_without_hostname, index=":")
        for i in file:
            virial_list = -voigt_6_to_full_3x3_stress(i.get_stress()) * i.get_volume()
            i.info["REF_virial"] = " ".join(map(str, virial_list.flatten()))
            del i.calc.results["stress"]
            i.arrays["REF_forces"] = i.calc.results["forces"]
            if atom_wise_regularization and (data_type == "phonon_dir"):
                atom_forces = np.array(i.arrays["REF_forces"])
                atom_wise_force = np.array(
                    [
                        force if force > f_min else f_min
                        for force in np.linalg.norm(atom_forces, axis=1)
                    ]
                )
                i.arrays["force_atom_sigma"] = regularization * atom_wise_force
            del i.calc.results["forces"]
            i.info["REF_energy"] = i.calc.results["free_energy"]
            del i.calc.results["energy"]
            del i.calc.results["free_energy"]
            i.info["config_type"] = config_type
            i.info["data_type"] = data_type.rstrip("_dir")
            i.pbc = True
        write("vasp_ref.extxyz", file, append=True)


class Species:
    """Species class."""

    def __init__(self, atoms):
        self.atoms = atoms

    def get_species(self):
        """
        Get species.

        Returns
        -------
        species_list:
            a list of species.
        """
        species_list = []

        for atom in self.atoms:
            symbol_all = atom.get_chemical_symbols()
            syms = list(set(symbol_all))
            species_list.extend(sym for sym in syms if sym not in species_list)

        return species_list

    def find_element_pairs(self, symbol_list=None):
        """
        Find element pairs.

        Parameters
        ----------
        symbol_list:
            list of symbols.

        Returns
        -------
        pairs:
            pairs of elements.

        """
        species_list = self.get_species() if symbol_list is None else symbol_list

        return list(combinations(species_list, 2))

    def get_number_of_species(self):
        """
        Get number of species.

        Returns
        -------
        number of species.

        """
        return int(len(self.get_species()))

    def get_species_Z(self):
        """
        Get species Z.

        Returns
        -------
        species_Z:
            species Z.
        """
        atom_numbers = []
        for atom_type in self.get_species():
            atom = Atoms(atom_type, [(0, 0, 0)])
            atom_numbers.append(int(atom.get_atomic_numbers()[0]))

        species_Z = "{"
        for i in range(len(atom_numbers) - 1):
            species_Z += str(atom_numbers[i]) + " "
        species_Z += str(atom_numbers[-1]) + "}"

        return species_Z


def flatten(atoms_object, recursive=False):
    """
    Flatten an iterable fully, but excluding Atoms objects.

    Parameters
    ----------
    atoms_object: Atoms object
    recursive: bool
        set the recursive boolean.

    Returns
    -------
    a flattened object, excluding the Atoms objects.

    """
    iteration_list = []

    if recursive:
        for element in atoms_object:
            if isinstance(element, Iterable) and not isinstance(
                element, (str, bytes, ase.atoms.Atoms, ase.Atoms)
            ):
                iteration_list.extend(flatten(element, recursive=True))
            else:
                iteration_list.append(element)
        return iteration_list

    return [item for sublist in atoms_object for item in sublist]


def gcm3_to_Vm(gcm3, mr, n_atoms=1):
    """
    Convert gcm3 to Vm.

    Parameters
    ----------
    gcm3:
        g/cm3
    mr:
    n_atoms:
        number of atoms.

    Returns
    -------
    the converted unit.

    """
    return 1 / (n_atoms * (gcm3 / mr) * 6.022e23 / (1e8) ** 3)


def get_atomic_numbers(species):
    """
    Get atomic numbers.

    Parameters
    ----------
    species:
        type of species

    Returns
    -------
    atomic_numbers:
        list of atomic numbers.

    """
    atom_numbers = []
    for atom_type in species:
        atom = Atoms(atom_type, [(0, 0, 0)])
        atom_numbers.append(int(atom.get_atomic_numbers()[0]))

    return atom_numbers


def split_dataset(atoms, split_ratio):
    """
    Split the dataset.

    Parameters
    ----------
    atoms: Atoms
        Ase atoms object
    split_ratio: float
        Parameter to divide the training set and the test set.

    Returns
    -------
    train_structures, test_structures:
        split-up datasets of train structures and test structures.

    """
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
    """
    For data distillation.

    Parameters
    ----------
    vasp_ref_dir:
        VASP reference data directory.
    f_max:
        maximally allowed force.

    Returns
    -------
    atoms_distilled:
        list of distilled atoms.

    """
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


def rms_dict(x_ref: np.ndarray, x_pred: np.ndarray) -> dict:
    """Compute RMSE and standard deviation of predictions with reference data.

    x_ref and x_pred should be of same shape.

    Parameters
    ----------
    x_ref : np.ndarray.
        list of reference data.
    x_pred: np.ndarray.
        list of prediction.

    Returns
    -------
    dict
        Dict with rmse and std deviation of predictions.
    """
    x_ref = np.array(x_ref)
    x_pred = np.array(x_pred)
    if np.shape(x_pred) != np.shape(x_ref):
        raise ValueError("WARNING: not matching shapes in rms")
    error_2 = (x_ref - x_pred) ** 2
    average = np.sqrt(np.average(error_2))
    std_ = np.sqrt(np.var(error_2))
    return {"rmse": average, "std": std_}


def energy_remain(in_file):
    """
    Plot the distribution of energy per atom on the output vs. the input.

    Parameters
    ----------
    in_file:
        input file

    Returns
    -------
    rms["rmse"]:
        distribution of energy per atom RMSE of output vs. input.

    """
    # read files
    in_atoms = ase.io.read(in_file, ":")
    ener_in = [
        at.info["REF_energy"] / len(at.get_chemical_symbols()) for at in in_atoms
    ]
    ener_out = [at.info["energy"] / len(at.get_chemical_symbols()) for at in in_atoms]
    rms = rms_dict(ener_in, ener_out)
    return rms["rmse"]


def extract_gap_label(xml_file_path):
    """
    Extract GAP label.

    Parameters
    ----------
    xml_file_path:
        path to the GAP fit potential xml file.

    Returns
    -------
    the extracted GAP label.

    """
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


def calculate_delta(atoms_db: list[Atoms], e_name: str) -> float:
    """
    Calculate delta parameter used for gap-fit.

    Parameters
    ----------
    atoms_db: list[Atoms]
        list of Ase atoms objects
    e_name: str
        energy_parameter_name as defined in gap-defaults.json

    Returns
    -------
    float
        delta parameter used for gap-fit (es_var / avg_neigh)

    """
    at_ids = [atom.get_atomic_numbers() for atom in atoms_db]
    isol_es = {
        atom.get_atomic_numbers()[0]: atom.info[e_name]
        for atom in atoms_db
        if "config_type" in atom.info and "isol" in atom.info["config_type"]
    }

    es_visol = np.array(
        [
            (atom.info[e_name] - sum([isol_es[j] for j in at_ids[ct]])) / len(atom)
            for ct, atom in enumerate(atoms_db)
        ]
    )
    es_var = np.var(es_visol)
    avg_neigh = np.mean([compute_average_coordination(atom) for atom in atoms_db])
    return es_var / avg_neigh


def compute_average_coordination(atoms: Atoms) -> float:
    """
    Compute average coordination.

    Parameters
    ----------
    atoms: Atoms
        Ase atoms object

    Returns
    -------
    float
        Average coordination - total_coordination / len(atoms)

    """
    cutoffs = natural_cutoffs(atoms)
    neighbor_list = NeighborList(cutoffs, self_interaction=False, bothways=True)
    neighbor_list.update(atoms)
    total_coordination = sum(
        len(neighbor_list.get_neighbors(index)[0]) for index in range(len(atoms))
    )
    return total_coordination / len(atoms)


def run_gap(num_processes: int, parameters):
    """
    GAP runner.

    num_processes: int
        number of threads to be used for the run.

    Parameters
    ----------
        GAP fit parameters.

    """
    os.environ["OMP_NUM_THREADS"] = str(num_processes)

    with open("std_gap_out.log", "w", encoding="utf-8") as file_std, open(
        "std_gap_err.log", "w", encoding="utf-8"
    ) as file_err:
        subprocess.call(["gap_fit", *parameters], stdout=file_std, stderr=file_err)


def run_quip(num_processes: int, data_path, xml_file: str, filename: str):
    """
    QUIP runner.

    num_processes: int
        number of threads to be used for the run.
    data_path:
        Path to the data file.
    filename: str
        Name of the output file.

    """
    os.environ["OMP_NUM_THREADS"] = str(num_processes)

    command = f"quip E=T F=T atoms_filename={data_path} param_filename={xml_file} | grep AT | sed 's/AT//' > {filename}"

    with open("std_quip_out.log", "w", encoding="utf-8") as file_std, open(
        "std_quip_err.log", "w", encoding="utf-8"
    ) as file_err:
        subprocess.call(command, stdout=file_std, stderr=file_err, shell=True)


def prepare_fit_environment(database_dir, mlip_path, glue_xml: bool):
    """
    Prepare the environment for the fit.

    Parameters
    ----------
    database_dir:
        Path to database directory.
    mlip_path:
        Path to the MLIP fit run (cwd).
    glue_xml: bool
            use the glue.xml core potential instead of fitting 2b terms.

    Returns
    -------
    the MLIP path.
    """
    if os.path.join(database_dir, "train_with_sigma.extxyz"):
        shutil.copy(
            os.path.join(database_dir, "train_with_sigma.extxyz"),
            os.path.join(mlip_path, "train_with_sigma.extxyz"),
        )
    shutil.copy(
        os.path.join(database_dir, "test.extxyz"),
        os.path.join(mlip_path, "test.extxyz"),
    )
    shutil.copy(
        os.path.join(database_dir, "train.extxyz"),
        os.path.join(mlip_path, "train.extxyz"),
    )
    if glue_xml:
        shutil.copy(
            os.path.join(database_dir, "../glue.xml"),  # very improvised on purpose
            os.path.join(mlip_path, "glue.xml"),
        )

    return mlip_path
