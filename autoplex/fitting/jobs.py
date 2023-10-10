"""
Jobs to fit ML potentials
"""
from __future__ import annotations

import numpy as np
from ase.io import read, write
import subprocess
from pathlib import Path
import re
import os
from jobflow import Flow, Response, job
from dataclasses import dataclass, field
from autoplex.fitting.utils import (
    load_gap_hyperparameter_defaults,
    gap_hyperparameter_constructor,
)

current_dir = Path(__file__).absolute().parent


@job
def gapfit(
    fit_input: dict,
    isolated_atoms: list,
    isolated_atoms_energy: list,
    path_to_default_hyperparameters: Path | str = current_dir / "gap-defaults.json",
    two_body: bool = True,
    three_body: bool = False,
    soap: bool = True,
    fit_kwargs: dict = field(default_factory=dict),
):
    """
    Prepares the GAP fit input and fits the data using GAP. More ML methods (e.g. ACE) to follow.

    Parameters
    ----------
    fit_input : dict.
        list containing static calculation directories.
    isolated_atoms : list.
        List of isolated atoms
    isolated_atoms_energy : list.
        List of isolated atoms energy
    path_to_default_hyperparameters : str or Path.
        Path to gap-defaults.json.
    two_body : bool.
        bool indicating whether to include two-body hyperparameters
    three_body : bool.
        bool indicating whether to include three-body hyperparameters
    soap : bool.
        bool indicating whether to include soap hyperparameters
    fit_kwargs : dict.
        dict including gap fit keyword args.

    Returns
    -------
    Response.output
        Path to the gap fit file.
    """

    flattened_input = lambda x: [
        y
        for z in x
        for y in (flattened_input(z) if isinstance(z, list) else [z])  # type:ignore
    ]
    fit = flattened_input(
        [
            dirs
            for data in fit_input.values()
            for datatype, dirs in data.items()
            if datatype != "phonon_data"
        ]
    )  # uniform data structure
    for entry in fit:
        file = read(re.sub(r"^.*?/", "/", entry, count=1) + "/OUTCAR.gz", index=":")
        for (
            i
        ) in (
            file
        ):  # credit goes to http://home.ustc.edu.cn/~lipai/scripts/ml_scripts/outcar2xyz.html
            xx, yy, zz, yz, xz, xy = -i.calc.results["stress"] * i.get_volume()
            i.info["virial"] = np.array([(xx, xy, xz), (xy, yy, yz), (xz, yz, zz)])
            del i.calc.results["stress"]
            i.pbc = True
        write("trainGAP.xyz", file, append=True)

    gap_default_hyperparameters = load_gap_hyperparameter_defaults(
        gap_fit_parameter_file_path=path_to_default_hyperparameters
    )

    e0: str = "{"

    for iso_atom, iso_energy in zip(isolated_atoms, isolated_atoms_energy):
        if iso_atom == isolated_atoms[-1]:
            e0 += str(iso_atom) + ":" + str(iso_energy) + "}"
        else:
            e0 += str(iso_atom) + ":" + str(iso_energy) + ":"
    # Updating the isolated atom energy
    gap_default_hyperparameters["general"].update({"e0": e0})
    # Overwriting the default gap_fit settings with user settings  # TODO XPOT support
    for key in gap_default_hyperparameters:
        for key2 in fit_kwargs:
            if key == key2:
                gap_default_hyperparameters[key].update(fit_kwargs[key2])

    gap = gap_hyperparameter_constructor(
        gap_parameter_dict=gap_default_hyperparameters,
        two_body=two_body,
        three_body=three_body,
        soap=soap,
    )

    general = [
        str(key) + "=" + str(gap_default_hyperparameters["general"][key])
        for key in gap_default_hyperparameters["general"]
    ]

    with open("std_out.log", "w") as file_std, open("std_err.log", "w") as file_err:
        subprocess.call(["gap_fit"] + general + [gap], stdout=file_std, stderr=file_err)

        directory = Path.cwd()

    return Response(
        output=str(
            os.path.join(directory, gap_default_hyperparameters["general"]["gp_file"])
        )
    )
