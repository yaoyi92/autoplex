"""fitting using GAP."""
from __future__ import annotations

import os
from pathlib import Path

import ase.io

from autoplex.fitting.common.utils import (
    calculate_delta,
    energy_remain,
    gap_hyperparameter_constructor,
    load_gap_hyperparameter_defaults,
    run_command,
)

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
) -> dict:
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
        gap_default_hyperparameters["general"].update({"at_file": train_data_path})
        if auto_delta:
            delta_3b = energy_remain("quip_train.extxyz")
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
        if auto_delta:
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
        f"export OMP_NUM_THREADS={num_processes} && quip E=T F=T atoms_filename={test_data_path} "
        f"param_filename=gap_file.xml | grep AT | sed 's/AT//' > quip_test.extxyz"
    )
    run_command(quip_command)
    test_error = energy_remain("quip_test.extxyz")
    print("Testing error of MLIP (eV/at.):", test_error)

    return {"train_error": train_error, "test_error": test_error}
