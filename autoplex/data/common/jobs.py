"""Jobs to create training data for ML potentials."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emmet.core.math import Matrix3D
    from pymatgen.core import Structure

import pickle

import matplotlib.pyplot as plt
import numpy as np
from ase.constraints import voigt_6_to_full_3x3_stress
from ase.io import read, write
from jobflow.core.job import job
from phonopy.structure.cells import get_supercell
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

from autoplex.data.common.utils import to_ase_trajectory


@job
def convert_to_extxyz(job_output, pkl_file, config_type, factor):
    """
    Convert data and write extxyt file.

    Parameters
    ----------
    job_output:
        the (static) job output object.
    pkl_file:
        a pickle file.
    config_type: str
            configuration type of the data.
    factor: str
            string of factor to resize cell parameters.

    """
    with open(pkl_file, "rb") as file:
        traj_obj = pickle.load(file)  # job_output.dir_name +
    # ForceFieldTaskDocument.from_ase_compatible_result() has no attribute dir_name implemented
    data = to_ase_trajectory(traj_obj=traj_obj)
    data[-1].write("tmp.xyz")
    file = read("tmp.xyz", index=":")
    for i in file:
        virial_list = -voigt_6_to_full_3x3_stress(i.get_stress()) * i.get_volume()
        i.info["REF_virial"] = " ".join(map(str, virial_list.flatten()))
        del i.calc.results["stress"]
        i.arrays["REF_forces"] = i.calc.results["forces"]
        del i.calc.results["forces"]
        i.info["REF_energy"] = i.calc.results["energy"]
        del i.calc.results["energy"]
        i.info["config_type"] = config_type
        i.pbc = True
    write("ref_" + factor + ".extxyz", file, append=True)


@job
def plot_force_distribution(
    cell_factor_sequence: list[float],
    x_min: int = 0,
    x_max: int = 5,
    bin_width: float = 0.125,
):
    """
    Plotter for the force distribution.

    Parameters
    ----------
    cell_factor_sequence: list[float]
        list of factor to resize cell parameters.
    x_min: int
        minimum value for the plot x-axis.
    x_max: int
        maximum value for the plot x-axis.
    bin_width: float
        width of the plot bins.

    """
    plt.xlabel("Forces")
    plt.ylabel("Count")
    bins = np.arange(x_min, x_max + bin_width, bin_width)
    plot_total = []
    for cell_factor in cell_factor_sequence:
        plot_data = []
        with open("ref_" + str(cell_factor).replace(".", "") + ".extxyz") as file:
            for line in file:
                # Split the line into columns
                columns = line.split()

                # Check if the line has exactly 10 columns
                if len(columns) == 10:
                    # Extract the last three columns
                    data = columns[-3:]
                    norm_data = np.linalg.norm(data, axis=-1)
                    plot_data.append(norm_data)

        plt.hist(plot_data, bins=bins, edgecolor="black")
        plt.title(f"Data for factor {cell_factor}")

        plt.savefig("Data_factor_" + str(cell_factor).replace(".", "") + ".png")
        plt.show()

        plot_total += plot_data
    plt.hist(plot_total, bins=bins, edgecolor="black")
    plt.title("Data")

    plt.savefig("Total_data.png")
    plt.show()


@job
def get_supercell_job(structure: Structure, supercell_matrix: Matrix3D):
    """
    Create a job to get the supercell.

    Parameters
    ----------
    structure: Structure
        pymatgen structure object.
    supercell_matrix: Matrix3D
        The matrix to generate the supercell.

    Returns
    -------
    supercell: Structure
        pymatgen structure object.

    """
    supercell = get_supercell(
        unitcell=get_phonopy_structure(structure), supercell_matrix=supercell_matrix
    )
    return get_pmg_structure(supercell)
