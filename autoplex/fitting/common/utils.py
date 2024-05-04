"""Utility functions for fitting jobs."""

from __future__ import annotations

import json
import os
import re
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
from ase.data import chemical_symbols
from ase.io import read, write
from ase.neighborlist import NeighborList, natural_cutoffs
from atomate2.utils.path import strip_hostname
from nequip.ase import NequIPCalculator
from scipy.spatial import ConvexHull
from scipy.special import comb
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
            delta_2b, num_triplet = calculate_delta(db_atoms, "REF_energy")
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
            delta_3b = delta_3b / num_triplet
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


def ace_fitting(
    db_dir: str | Path,
    order: int = 4,
    totaldegree: int = 16,
    cutoff: float = 5.0,
    solver: str = "BLR",
    isol_es: dict | None = None,
    num_processes: int = 32,
):
    """
    Perform the ACE (Atomic Cluster Expansion) potential fitting.

    This function sets up and executes a Julia script to perform ACE fitting using specified parameters
    and input data located in the provided directory. It handles the input/output of atomic configurations,
    sets up the ACE model, and calculates training and testing errors after fitting.

    Parameters
    ----------
    db_dir: str or Path
        directory containing the training and testing data files.
    order: int
        order of ACE.
    totaldegree: int
        total degree of the polynomial terms in the ACE model.
    cutoff: float
        cutoff distance for atomic interactions in the ACE model.
    solver: str
        solver to be used for fitting the ACE model. Default is "BLR" (Bayesian Linear Regression).
        For very large-scale parameter estimation problems, using "LSQR" solver.
    isol_es: dict:
        mandatory dictionary mapping element numbers to isolated energies.
    num_processes: int
        number of processes to use for parallel computation.

    Returns
    -------
    dict[str, float]
        A dictionary containing train_error, test_error, and the path to the fitted MLIP.

    Raises
    ------
    - ValueError: If the `isol_es` dictionary is empty or not provided when required.

    Example:
    >>> result = ace_fitting('/path/to/data', order=2, totaldegree=12, cutoff=6.0, solver='BLR', num_processes=4)
    >>> print(result['train_error'], result['test_error'])
    """
    train_atoms = ase.io.read(os.path.join(db_dir, "train.extxyz"), index=":")
    source_file_path = os.path.join(db_dir, "test.extxyz")
    shutil.copy(source_file_path, ".")
    isol_es_update = {}

    if isol_es:
        for e_num, e_energy in isol_es.items():
            isol_es_update[chemical_symbols[int(e_num)]] = e_energy
    else:
        raise ValueError("isol_es is empty or not defined!")

    formatted_isol_es = (
        "["
        + ", ".join([f":{key} => {value}" for key, value in isol_es_update.items()])
        + "]"
    )
    formatted_species = (
        "[" + ", ".join([f":{key}" for key, value in isol_es_update.items()]) + "]"
    )

    train_ace = [
        at for at in train_atoms if "IsolatedAtom" not in at.info["config_type"]
    ]
    ase.io.write("train_ace.extxyz", train_ace, format="extxyz")

    ace_text = f"""using ACEpotentials
using LinearAlgebra: norm, Diagonal
using CSV, DataFrames
using Distributed
addprocs({num_processes-1}, exeflags="--project=$(Base.active_project())")
@everywhere using ACEpotentials

data_file = "train_ace.extxyz"
data = read_extxyz(data_file)
test_data_file = "test.extxyz"
test_data = read_extxyz(test_data_file)
data_keys = (energy_key = "REF_energy", force_key = "REF_force", virial_key = "REF_virial")

model = acemodel(elements={formatted_species},
                order={order},
                totaldegree={totaldegree},
                rcut={cutoff},
                Eref={formatted_isol_es})

weights = Dict(
            "crystal" => Dict("E" => 30.0, "F" => 1.0 , "V" => 1.0 ),
            "RSS" => Dict("E" => 3.0, "F" => 0.5 , "V" => 0.1 ),
            "amorphous" => Dict("E" => 3.0, "F" => 0.5 , "V" => 0.1 ),
            "liquid" => Dict("E" => 10.0, "F" => 0.5 , "V" => 0.25 ),
            "RSS_initial" => Dict("E" => 1.0, "F" => 0.5 , "V" => 0.1 ),
            "dimer" => Dict("E" => 30.0, "F" => 1.0 , "V" => 1.0 )
            )

P = smoothness_prior(model; p = 4)

solver = ACEfit.{solver}()

acefit!(model, data; solver=solver, weights=weights, prior = P, data_keys...)

@info("Training Error Table")
ACEpotentials.linear_errors(data, model; data_keys...)

@info("Testing Error Table")
ACEpotentials.linear_errors(test_data, model; data_keys...)

@info("Manual RMSE Test")
potential = model.potential
train_energies = [ JuLIP.get_data(at, "REF_energy") / length(at) for at in data]
model_energies_train = [energy(potential, at) / length(at) for at in data]
rmse_energy_train = norm(train_energies - model_energies_train) / sqrt(length(data))
test_energies = [ JuLIP.get_data(at, "REF_energy") / length(at) for at in test_data]
model_energies_pred = [energy(potential, at) / length(at) for at in test_data]
rmse_energy_test = norm(test_energies - model_energies_pred) / sqrt(length(test_data))

df = DataFrame(rmse_energy_train = rmse_energy_train, rmse_energy_test = rmse_energy_test)
CSV.write("rmse_energies.csv", df)

save_potential("acemodel.json", model)
export2lammps("acemodel.yace", model)
    """

    with open("ace.jl", "w") as file:
        file.write(ace_text)

    os.system(f"export OMP_NUM_THREADS={num_processes} && julia ace.jl")

    energy_err = pd.read_csv("rmse_energies.csv")
    train_error = energy_err["rmse_energy_train"][0]
    test_error = energy_err["rmse_energy_test"][0]

    return {
        "train_error": train_error,
        "test_error": test_error,
        "mlip_path": Path.cwd(),
    }


def nequip_fitting(
    db_dir: str,
    r_max: float = 4.0,
    num_layers: int = 4,
    l_max: int = 2,
    num_features: int = 32,
    num_basis: int = 8,
    invariant_layers: int = 2,
    invariant_neurons: int = 64,
    batch_size: int = 5,
    learning_rate: float = 0.005,
    max_epochs: int = 10000,
    default_dtype: str = "float32",
    isol_es: dict | None = None,
    device: str = "cuda",
):
    """
    Perform the NequIP potential fitting.

    This function sets up and executes a python script to perform NequIP fitting using specified parameters
    and input data located in the provided directory. It handles the input/output of atomic configurations,
    sets up the NequIP model, and calculates training and testing errors after fitting.

    Parameters
    ----------
    db_dir: str or Path
        directory containing the training and testing data files.
    r_max: float
        cutoff radius in length units
    num_layers: int
        number of interaction blocks
    l_max: int
        maximum irrep order (rotation order) for the network's features
    num_features: int
        multiplicity of the features
    num_basis: int
        number of basis functions used in the radial basis
    invariant_layers: int
        number of radial layers
    invariant_neurons: int
        number of hidden neurons in radial function
    batch_size: int
        batch size
    learning_rate: float
        learning rate
    default_dtype: str
        type of float to use, e.g. float32 and float64
    isol_es: dict
        mandatory dictionary mapping element numbers to isolated energies.
    device: str
        specify device to use cuda or cpu

    Returns
    -------
    dict[str, float]
        A dictionary containing train_error, test_error, and the path to the fitted MLIP.

    Raises
    ------
    - ValueError: If the `isol_es` dictionary is empty or not provided when required.
    """
    train_data = ase.io.read(os.path.join(db_dir, "train.extxyz"), index=":")
    train_nequip = [
        at for at in train_data if "IsolatedAtom" not in at.info["config_type"]
    ]
    ase.io.write("train_nequip.extxyz", train_nequip, format="extxyz")

    test_data = ase.io.read(os.path.join(db_dir, "test.extxyz"), index=":")
    num_of_train = len(train_nequip)
    num_of_val = len(test_data)

    isol_es_update = ""
    ele_syms = []
    if isol_es:
        for e_num in isol_es:
            ele_sym = "  - " + chemical_symbols[int(e_num)] + "\n"
            isol_es_update += ele_sym
            ele_syms.append(chemical_symbols[int(e_num)])
    else:
        raise ValueError("isol_es is empty or not defined!")

    nequip_text = f"""root: results
run_name: autoplex
seed: 123
dataset_seed: 456
append: true
default_dtype: {default_dtype}

# network
r_max: {r_max}
num_layers: {num_layers}
l_max: {l_max}
parity: true
num_features: {num_features}
nonlinearity_type: gate

nonlinearity_scalars:
  e: silu
  o: tanh

nonlinearity_gates:
  e: silu
  o: tanh

num_basis: {num_basis}
BesselBasis_trainable: true
PolynomialCutoff_p: 6

invariant_layers: {invariant_layers}
invariant_neurons: {invariant_neurons}
avg_num_neighbors: auto

use_sc: true
dataset: ase
validation_dataset: ase
dataset_file_name: ./train_nequip.extxyz
validation_dataset_file_name: {db_dir}/test.extxyz

ase_args:
  format: extxyz
dataset_key_mapping:
  REF_energy: total_energy
  REF_forces: forces
validation_dataset_key_mapping:
  REF_energy: total_energy
  REF_forces: forces

chemical_symbols:
{isol_es_update}
wandb: False
wandb_project: autoplex

verbose: info
log_batch_freq: 10
log_epoch_freq: 1
save_checkpoint_freq: -1
save_ema_checkpoint_freq: -1

n_train: {num_of_train}
n_val: {num_of_val}
learning_rate: {learning_rate}
batch_size: {batch_size}
validation_batch_size: 10
max_epochs: {max_epochs}
shuffle: true
metrics_key: validation_loss
use_ema: true
ema_decay: 0.99
ema_use_num_updates: true
report_init_validation: true

early_stopping_patiences:
  validation_loss: 50

early_stopping_lower_bounds:
  LR: 1.0e-5

loss_coeffs:
  forces: 1
  total_energy:
    - 1
    - PerAtomMSELoss

metrics_components:
  - - forces
    - mae
  - - forces
    - rmse
  - - forces
    - mae
    - PerSpecies: True
      report_per_component: False
  - - forces
    - rmse
    - PerSpecies: True
      report_per_component: False
  - - total_energy
    - mae
  - - total_energy
    - mae
    - PerAtom: True

optimizer_name: Adam
optimizer_amsgrad: true

lr_scheduler_name: ReduceLROnPlateau
lr_scheduler_patience: 100
lr_scheduler_factor: 0.5

per_species_rescale_shifts_trainable: false
per_species_rescale_scales_trainable: false

per_species_rescale_shifts: dataset_per_atom_total_energy_mean
per_species_rescale_scales: dataset_forces_rms
    """

    with open("nequip.yaml", "w") as file:
        file.write(nequip_text)

    run_nequip("nequip-train nequip.yaml", "nequip_train")
    run_nequip(
        "nequip-deploy build --train-dir results/autoplex ./deployed_nequip_model.pth",
        "nequip_deploy",
    )

    calc = NequIPCalculator.from_deployed_model(
        model_path="deployed_nequip_model.pth",
        device=device,
        species_to_type_name={s: s for s in ele_syms},
        set_global_options=False,
    )

    ener_out_train = []
    for at in train_nequip:
        at.calc = calc
        ener_out_train.append(at.get_potential_energy() / len(at))

    ener_in_train = [
        at.info["REF_energy"] / len(at.get_chemical_symbols()) for at in train_nequip
    ]

    train_error = rms_dict(ener_in_train, ener_out_train)["rmse"]

    ener_out_test = []
    for at in test_data:
        at.calc = calc
        ener_out_test.append(at.get_potential_energy() / len(at))

    ener_in_test = [
        at.info["REF_energy"] / len(at.get_chemical_symbols()) for at in test_data
    ]

    test_error = rms_dict(ener_in_test, ener_out_test)["rmse"]

    return {
        "train_error": train_error,
        "test_error": test_error,
        "mlip_path": Path.cwd(),
    }


def mace_fitting(
    db_dir: str,
    model: str = "MACE",
    config_type_weights: str = None,
    hidden_irreps: str = None,
    r_max: float = 4.0,
    batch_size: int = 10,
    max_num_epochs: int = 1000,
    start_swa: str = None,
    ema_decay: str = None,
    correlation: int = 3,
    loss: str = None,
    default_dtype: str = None,
    device: str = "cuda",
):
    """
    Perform the MACE potential fitting.

    This function sets up and executes a python script to perform MACE fitting using specified parameters
    and input data located in the provided directory. It handles the input/output of atomic configurations,
    sets up the NequIP model, and calculates training and testing errors after fitting.

    Parameters
    ----------
    db_dir: str or Path
        directory containing the training and testing data files.
    model: str
        type of model to be trained
    config_type_weights: str
        weights of config types
    hidden_irreps: str
        control the model size
    r_max: float
        cutoff radius controls the locality of the model
    batch_size: int
        batch size (note that batch size cannot be larger than the size of training datasets)
    start_swa: str
        if the keyword --swa is enabled, the energy weight of the loss is increased
        for the last ~20% of the training epochs (from --start_swa epochs)
    correlation: int
        correlation order corresponds to the order that MACE induces at each layer
    loss: str
        loss functions
    default_dtype: str
        type of float to use, e.g. float32 and float64
    device: str
        specify device to use cuda or cpu

    Returns
    -------
    dict[str, float]
        A dictionary containing train_error, test_error, and the path to the fitted MLIP.

    """
    hypers = [
        "--name=MACE_model",
        f"--train_file={db_dir}/train.extxyz",
        f"--valid_file={db_dir}/test.extxyz",
        f"--config_type_weights={config_type_weights}",
        f"--model={model}",
        f"--hidden_irreps={hidden_irreps}",
        "--energy_key=REF_energy",
        "--forces_key=REF_forces",
        f"--r_max={r_max}",
        f"--correlation={correlation}",
        f"--batch_size={batch_size}",
        f"--max_num_epochs={max_num_epochs}",
        "--swa",
        f"--start_swa={start_swa}",
        "--ema",
        f"--ema_decay={ema_decay}",
        "--amsgrad",
        f"--loss={loss}",
        "--restart_latest",
        "--seed=12345",
        f"--default_dtype={default_dtype}",
        f"--device={device}",
    ]

    run_mace(hypers)

    with open("./logs/MACE_model_run-12345.log") as file:
        log_data = file.read()

    tables = re.split(r"\+-+\+\n", log_data)
    if tables:
        last_table = tables[-2]
        matches = re.findall(r"\|\s*(train|valid)\s*\|\s*([\d\.]+)\s*\|", last_table)

    return {
        "train_error": float(matches[0][1]),
        "test_error": float(matches[1][1]),
        "mlip_path": Path.cwd(),
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
            and at.info["config_type"] != "IsolatedAtom"
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


def calculate_delta(atoms_db: list[Atoms], e_name: str) -> tuple[float, float]:
    """
    Calculate the delta parameter and average number of triplets for gap-fitting.

    Parameters
    ----------
    atoms_db: list[Atoms]
        list of Ase atoms objects
    e_name: str
        energy_parameter_name as defined in gap-defaults.json

    Returns
    -------
    tuple[float, float]
        A tuple containing:
        - delta parameter used for gap-fit, calculated as (es_var / avg_neigh).
        - Average number of triplets per atom.

    """
    at_ids = [atom.get_atomic_numbers() for atom in atoms_db]
    isol_es = {
        atom.get_atomic_numbers()[0]: atom.info[e_name]
        for atom in atoms_db
        if "config_type" in atom.info and "IsolatedAtom" in atom.info["config_type"]
    }

    es_visol = np.array(
        [
            (atom.info[e_name] - sum([isol_es[j] for j in at_ids[ct]])) / len(atom)
            for ct, atom in enumerate(atoms_db)
        ]
    )
    es_var = np.var(es_visol)
    avg_neigh = np.mean([compute_pairs_triplets(atom)[0] for atom in atoms_db])
    num_triplet = np.mean([compute_pairs_triplets(atom)[1] for atom in atoms_db])

    return es_var / avg_neigh, num_triplet


def compute_pairs_triplets(atoms):
    """
    Calculate the number of pairwise and triplet within a cutoff distance for a given list of atoms.

    Parameters
    ----------
    atoms : ASE atoms object

    Returns
    -------
    list[float, float]
        Returns a list of the number of pairs or triplets an atom is involved in.

    """
    cutoffs = natural_cutoffs(atoms)
    neighbor_list = NeighborList(
        cutoffs=cutoffs, skin=0.15, self_interaction=False, bothways=True
    )
    neighbor_list.update(atoms)
    counts_list = [
        len(neighbor_list.get_neighbors(index)[0]) for index in range(len(atoms))
    ]
    num_pair = sum(counts_list) / len(atoms)

    triplets = [comb(count, 2) for count in counts_list if count > 1]
    num_triplet = sum(triplets) / len(atoms)

    return [num_pair, num_triplet]


def run_ace(num_processes: int, script_name: str):
    """
    Julia-ACE script runner.

    Parameters
    ----------
    num_processes: int
        Number of threads to be used for the run.
    script_name: str
        Name of the Julia script to run.

    """
    os.environ["JULIA_NUM_THREADS"] = str(num_processes)

    with open("julia-ace_out.log", "w", encoding="utf-8") as file_out, open(
        "julia-ace_err.log", "w", encoding="utf-8"
    ) as file_err:
        subprocess.call(["julia", script_name], stdout=file_out, stderr=file_err)


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


def run_nequip(command: str, log_prefix: str):
    """
    Nequip runner.

    Parameters
    ----------
    command: str
        The command to execute, along with its arguments.
    log_prefix: str
        Prefix for log file names, used to differentiate between different commands' logs.

    """
    with open(f"{log_prefix}_out.log", "w", encoding="utf-8") as file_out, open(
        f"{log_prefix}_err.log", "w", encoding="utf-8"
    ) as file_err:
        subprocess.call(command.split(), stdout=file_out, stderr=file_err)


def run_mace(hypers: list):
    """
    MACE runner.

    Parameters
    ----------
    hypers: list
        containing all hyperparameters required for the MACE model training.

    """
    with open("mace_train_out.log", "w", encoding="utf-8") as file_std, open(
        "mace_train_err.log", "w", encoding="utf-8"
    ) as file_err:
        subprocess.call(["mace_run_train", *hypers], stdout=file_std, stderr=file_err)


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
