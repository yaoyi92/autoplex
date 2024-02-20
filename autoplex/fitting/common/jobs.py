"""fitting using GAP."""
from __future__ import annotations

import os
from pathlib import Path

import ase.io
import numpy as np
from ase.neighborlist import NeighborList, natural_cutoffs

from autoplex.fitting.common.utils import (
    energy_remain,
    gap_hyperparameter_constructor,
    load_gap_hyperparameter_defaults,
)

current_dir = Path(__file__).absolute().parent
GAP_DEFAULTS_FILE_PATH = current_dir / "gap-defaults.json"


def calculate_delta(atoms_db, e_name):
    """
    Calculate delta.

    Parameters
    ----------
    atoms_db
    e_name

    Returns
    -------
    es_var / avg_neigh

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


def compute_average_coordination(atoms):
    """
    Compute average coordination.

    Parameters
    ----------
    atoms

    Returns
    -------
    total_coordination / len(atoms)

    """
    cutoffs = natural_cutoffs(atoms)
    neighbor_list = NeighborList(cutoffs, self_interaction=False, bothways=True)
    neighbor_list.update(atoms)
    total_coordination = sum(
        len(neighbor_list.get_neighbors(index)[0]) for index in range(len(atoms))
    )
    return total_coordination / len(atoms)


def run_command(command):
    """
    Run command.

    Parameters
    ----------
    command

    Returns
    -------
    os.system(command)

    """
    os.system(command)


def gap_fitting(
    db_dir: str | Path,
    include_two_body: bool = True,
    include_three_body: bool = False,
    include_soap: bool = True,
    path_to_default_hyperparameters: Path | str = GAP_DEFAULTS_FILE_PATH,
    num_processes: int = 32,
    fit_kwargs: dict | None = None,  # pylint: disable=E3701
):
    """
    GAP fit and validation job.

    Parameters
    ----------
    db_dir: str or path.
        Path to database directory
    path_to_default_hyperparameters : str or Path.
        Path to gap-defaults.json.
    include_two_body : bool.
        bool indicating whether to include two-body hyperparameters
    include_three_body : bool.
        bool indicating whether to include three-body hyperparameters
    include_soap : bool.
        bool indicating whether to include soap hyperparameters
    fit_kwargs: dict.
        optional dictionary with parameters for gap fitting with keys same as
        gap-defaults.json.

    Returns
    -------
    train_error, test_error

    """
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

    if include_two_body:
        delta_2b = calculate_delta(db_atoms, "REF_energy")
        gap_default_hyperparameters["general"].update({"at_file": train_data_path})
        gap_default_hyperparameters["twob"].update({"delta": delta_2b})

        fit_parameters_list = gap_hyperparameter_constructor(
            gap_parameter_dict=gap_default_hyperparameters,
            include_two_body=include_two_body,
        )
        fit_parameters_string = " ".join(fit_parameters_list)

        gap_command = (
            f"export OMP_NUM_THREADS={num_processes} && "
            f"gap_fit {fit_parameters_string}"
        )
        run_command(gap_command)

        quip_command = (
            f"export OMP_NUM_THREADS={num_processes} && quip E=T F=T atoms_filename={train_data_path} "
            f"param_filename=gap_file.xml | grep AT | sed 's/AT//' > quip_train.extxyz"
        )
        run_command(quip_command)

    if include_three_body:
        delta_3b = energy_remain("quip_train.extxyz")
        gap_default_hyperparameters["general"].update({"at_file": train_data_path})
        gap_default_hyperparameters["threeb"].update({"delta": delta_3b})

        fit_parameters = gap_hyperparameter_constructor(
            gap_parameter_dict=gap_default_hyperparameters,
            include_two_body=include_two_body,
            include_three_body=include_three_body,
        )
        fit_parameters_string = " ".join(fit_parameters)

        gap_command = (
            f"export OMP_NUM_THREADS={num_processes} && "
            f"gap_fit {fit_parameters_string}"
        )
        run_command(gap_command)

        quip_command = (
            f"export OMP_NUM_THREADS={num_processes} && quip E=T F=T atoms_filename={train_data_path} "
            f"param_filename=gap_file.xml | grep AT | sed 's/AT//' > quip_train.extxyz"
        )
        run_command(quip_command)

    if include_soap:
        delta_soap = (
            energy_remain("quip_train.extxyz")
            if include_two_body or include_three_body
            else 1
        )

        gap_default_hyperparameters["general"].update({"at_file": train_data_path})
        gap_default_hyperparameters["soap"].update({"delta": delta_soap})

        fit_parameters = gap_hyperparameter_constructor(
            gap_parameter_dict=gap_default_hyperparameters,
            include_two_body=include_two_body,
            include_three_body=include_three_body,
            include_soap=include_soap,
        )
        fit_parameters_string = " ".join(fit_parameters)

        gap_command = (
            f"export OMP_NUM_THREADS={num_processes} && "
            f"gap_fit {fit_parameters_string}"
        )
        run_command(gap_command)

        quip_command = (
            f"export OMP_NUM_THREADS={num_processes} && quip E=T F=T atoms_filename={train_data_path} "
            f"param_filename=gap_file.xml | grep AT | sed 's/AT//' > quip_train.extxyz"
        )
        run_command(quip_command)

    # Calculate training error
    train_error = energy_remain("quip_train.extxyz")
    print("Training error of MLIP (eV/at.):", train_error)

    # Calculate testing error
    quip_command = (
        f"export OMP_NUM_THREADS=32 && quip E=T F=T atoms_filename={test_data_path} "
        f"param_filename=gap_file.xml | grep AT | sed 's/AT//' > quip_test.extxyz"
    )
    run_command(quip_command)
    test_error = energy_remain("quip_test.extxyz")
    print("Testing error of MLIP (eV/at.):", test_error)

    return train_error, test_error
