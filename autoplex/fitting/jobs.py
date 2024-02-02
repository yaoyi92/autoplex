"""Jobs to fit ML potentials."""
from __future__ import annotations

import os
import subprocess
from dataclasses import field
from pathlib import Path

from jobflow import Response, job

from autoplex.fitting.utils import (
    gap_hyperparameter_constructor,
    get_list_of_vasp_calc_dirs,
    load_gap_hyperparameter_defaults,
    outcar_2_extended_xyz,
)

current_dir = Path(__file__).absolute().parent
GAP_DEFAULTS_FILE_PATH = current_dir / "gap-defaults.json"


@job
def gapfit(
    fit_input: dict,
    isolated_atoms: list,
    isolated_atoms_energy: list,
    path_to_default_hyperparameters: Path | str = GAP_DEFAULTS_FILE_PATH,
    include_two_body: bool = True,
    include_three_body: bool = False,
    include_soap: bool = True,
    xyz_file: str | None = None,
    fit_kwargs=None,  # pylint: disable=E3701
):  # pylint: disable=R0913, R0914
    """
    Prepare the GAP fit input and fits the data using GAP. More ML methods (e.g. ACE) to follow.

    Parameters
    ----------
    fit_input : dict.
        PhononDFTMLDataGenerationFlow output.
    isolated_atoms : list.
        List of element names (str) for computation of isolated atoms.
    isolated_atoms_energy : list.
        List of isolated atoms energy
    path_to_default_hyperparameters : str or Path.
        Path to gap-defaults.json.
    include_two_body : bool.
        bool indicating whether to include two-body hyperparameters
    include_three_body : bool.
        bool indicating whether to include three-body hyperparameters
    include_soap : bool.
        bool indicating whether to include soap hyperparameters
    xyz_file: str or None
        a possibly already existing xyz file
    config_types: list[str] or None
            list of config_types.
    fit_kwargs : dict.
        dict including gap fit keyword args.

    Returns
    -------
    Response.output
        Path to the gap fit file.
    """
    config_types = []
    if fit_kwargs is None:
        fit_kwargs = field(default_factory=dict)

    list_of_vasp_calc_dirs = get_list_of_vasp_calc_dirs(flow_output=fit_input)

    config_types = [
        key
        for key, value in fit_input.items()
        for key2, value2 in value.items()
        if key2 != "phonon_data"
        for _ in value2[0]
    ]

    outcar_2_extended_xyz(
        path_to_vasp_static_calcs=list_of_vasp_calc_dirs,
        config_types=config_types,
        xyz_file=xyz_file,
    )

    gap_default_hyperparameters = load_gap_hyperparameter_defaults(
        gap_fit_parameter_file_path=path_to_default_hyperparameters
    )

    # Update the default gap_fit settings with user provided settings  # TODO XPOT support
    for parameter in gap_default_hyperparameters:
        for arg in fit_kwargs:
            if parameter == arg:
                gap_default_hyperparameters[parameter].update(fit_kwargs[arg])

    # check and update args of gap_hyperparameter_constructor
    for arg in fit_kwargs:
        if arg == "include_two_body":
            include_two_body = fit_kwargs[arg]
        elif arg == "include_three_body":
            include_three_body = fit_kwargs[arg]
        elif arg == "include_soap":
            include_soap = fit_kwargs[arg]

    gap_parameters = gap_hyperparameter_constructor(
        gap_parameter_dict=gap_default_hyperparameters,
        atoms_symbols=isolated_atoms,
        atoms_energies=isolated_atoms_energy,
        include_two_body=include_two_body,
        include_three_body=include_three_body,
        include_soap=include_soap,
    )

    with open("std_out.log", "w", encoding="utf-8") as file_std, open(
        "std_err.log", "w", encoding="utf-8"
    ) as file_err:
        subprocess.call(["gap_fit", *gap_parameters], stdout=file_std, stderr=file_err)

        directory = Path.cwd()

    return Response(
        output=str(
            os.path.join(directory, gap_default_hyperparameters["general"]["gp_file"])
        )
    )
