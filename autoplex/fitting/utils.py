"""Utility functions for fitting jobs"""

from __future__ import annotations

import json
from pathlib import Path


def load_gap_hyperparameter_defaults(gap_fit_parameter_file_path: str | Path):
    """
    Loads gap fit default parameters from the json file

    Parameters
    ----------
    gap_fit_parameter_file_path : str or Path.
        Path to gap-defaults.json.

    Returns
    -------
    dict
       gap fit default parameters.
    """
    with open(gap_fit_parameter_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data


def gap_hyperparameter_constructor(
    gap_parameter_dict: dict,
    two_body: bool = True,
    three_body: bool = False,
    soap: bool = True,
):
    """
    Constructs a string with gap fit hyperparameters from the parameters dict

    Parameters
    ----------
    gap_parameter_dict : dict.
        dictionary with gap hyperparameters.
    two_body : bool.
        bool indicating whether to include two-body hyperparameters
    three_body : bool.
        bool indicating whether to include three-body hyperparameters
    soap : bool.
        bool indicating whether to include soap hyperparameters

    Returns
    -------
        str
           gap fit parameter string.
    """
    two_body_params = " ".join(
        [
            f"{key}={value}"
            for key, value in gap_parameter_dict["twob"].items()
            if two_body is True
        ]
    )
    three_body_params = " ".join(
        [
            f"{key}={value}"
            for key, value in gap_parameter_dict["threeb"].items()
            if three_body is True
        ]
    )
    soap_params = str(":soap " if soap is True else "") + " ".join(
        [
            f"{key}={value}"
            for key, value in gap_parameter_dict["soap"].items()
            if soap is True
        ]
    )

    gap = f"gap={{{two_body_params} {three_body_params} {soap_params}}}"

    return gap
