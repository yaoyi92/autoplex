"""Jobs to create training data for ML potentials."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emmet.core.math import Matrix3D

import pickle
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from ase.constraints import voigt_6_to_full_3x3_stress
from ase.io import read, write
from jobflow.core.job import job
from phonopy.structure.cells import get_supercell
from pymatgen.core import Structure
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure

from autoplex.data.common.utils import (
    mc_rattle,
    random_vary_angle,
    scale_cell,
    std_rattle,
    to_ase_trajectory,
)


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


@job
def generate_randomized_structures(
    structure: Structure,
    distort_type: int = 0,
    n_structures: int = 10,
    volume_scale_factor_range: list[float] | None = None,
    volume_custom_scale_factors: list[float] | None = None,
    min_distance: float = 1.5,
    angle_percentage_scale: float = 10,
    angle_max_attempts: int = 1000,
    rattle_type: int = 0,
    rattle_std: float = 0.01,
    rattle_seed: int = 42,
    rattle_mc_n_iter: int = 10,
    w_angle: list[float] | None = None,
):
    """
    Take in a pymatgen Structure object and generates angle/volume distorted + rattled structures.

    Parameters
    ----------
    structure : Structure.
        Pymatgen structures object.
    distort_type : int.
        0- volume distortion, 1- angle distortion, 2- volume and angle distortion. Default=0.
    n_structures : int.
        Total number of distorted structures to be generated.
        Must be provided if distorting volume without specifying a range, or if distorting angles.
        Default=10.
    volume_scale_factor_range : list[float]
        [min, max] of volume scale factors.
        e.g. [0.90, 1.10] will distort volume +-10%.
    volume_custom_scale_factors : list[float]
        Specify explicit scale factors (if range is not specified).
        If None, will default to [0.90, 0.95, 0.98, 0.99, 1.01, 1.02, 1.05, 1.10].
    min_distance: float
        Minimum separation allowed between any two atoms.
        Default= 1.5A.
    angle_percentage_scale: float
        Angle scaling factor.
        Default= 10 will randomly distort angles by +-10% of original value.
    angle_max_attempts: int.
        Maximum number of attempts to distort structure before aborting.
        Default=1000.
    w_angle: list[float]
        List of angle indices to be changed i.e. 0=alpha, 1=beta, 2=gamma.
        Default= [0, 1, 2].
    rattle_type: int.
        0- standard rattling, 1- Monte-Carlo rattling. Default=0.
    rattle_std: float.
        Rattle amplitude (standard deviation in normal distribution).
        Default=0.01.
        Note that for MC rattling, displacements generated will roughly be
        rattle_mc_n_iter**0.5 * rattle_std for small values of n_iter.
    rattle_seed: int.
        Seed for setting up NumPy random state from which random numbers are generated.
        Default=42.
    rattle_mc_n_iter: int.
        Number of Monte Carlo iterations.
        Larger number of iterations will generate larger displacements.
        Default=10.

    Returns
    -------
    Response.output.
        Volume or angle-distorted structures with rattled atoms.
    """
    # distort cells by volume or angle
    if distort_type == 0:
        distorted_cells = scale_cell(
            structure=structure,
            volume_scale_factor_range=volume_scale_factor_range,
            n_structures=n_structures,
            volume_custom_scale_factors=volume_custom_scale_factors,
        )
    elif distort_type == 1:
        distorted_cells = random_vary_angle(
            structure=Structure,
            min_distance=min_distance,
            angle_percentage_scale=angle_percentage_scale,
            w_angle=w_angle,
            n_structures=n_structures,
            angle_max_attempts=angle_max_attempts,
        )
    elif distort_type == 2:
        initial_distorted_cells = scale_cell(
            structure=structure,
            volume_scale_factor_range=volume_scale_factor_range,
            n_structures=n_structures,
            volume_custom_scale_factors=volume_custom_scale_factors,
        )
        distorted_cells = []
        for cell in initial_distorted_cells:
            distorted_cell = random_vary_angle(
                structure=cell,
                min_distance=min_distance,
                angle_percentage_scale=angle_percentage_scale,
                w_angle=w_angle,
                n_structures=1,
                angle_max_attempts=angle_max_attempts,
            )
            distorted_cells.append(distorted_cell)
    else:
        raise TypeError("distort_type is not recognised")

    # rattle cells by standard or mc
    rattled_cells = (
        [
            std_rattle(
                structure=cell,
                n_structures=1,
                rattle_std=rattle_std,
                rattle_seed=rattle_seed,
            )
            for cell in distorted_cells
        ]
        if rattle_type == 0
        else [
            mc_rattle(
                structure=cell,
                n_structures=1,
                rattle_std=rattle_std,
                min_distance=min_distance,
                rattle_seed=rattle_seed,
                rattle_mc_n_iter=rattle_mc_n_iter,
            )
            for cell in distorted_cells
        ]
        if rattle_type == 1
        else None
    )

    if rattled_cells is None:
        raise TypeError("rattle_type is not recognized")

    return list(chain.from_iterable(rattled_cells))
