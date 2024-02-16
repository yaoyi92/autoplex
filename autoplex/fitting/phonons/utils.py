"""Utility functions for fitting jobs."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from ase.constraints import voigt_6_to_full_3x3_stress
from ase.io import read, write
from atomate2.utils.path import strip_hostname


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
    include_two_body: bool = True,
    include_three_body: bool = False,
    include_soap: bool = True,
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
