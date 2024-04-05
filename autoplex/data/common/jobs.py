"""Jobs to create training data for ML potentials."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emmet.core.math import Matrix3D
    from pymatgen.core import Structure

import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
from ase.constraints import voigt_6_to_full_3x3_stress
from ase.io import read, write
from hiphive.structure_generation import generate_mc_rattled_structures
from jobflow.core.job import job
from phonopy.structure.cells import get_supercell
from pymatgen.io.ase import AseAtomsAdaptor
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


@job
def generate_randomized_structures(
    # leaving this here and adding the duplicate in common to avoid the respective unit tests from failing
    structure: Structure,
    n_struct: int,
    cell_factor_sequence: list[float] | None = None,
    std_dev: float = 0.01,
):
    """
    Take in a pymatgen Structure object and generates randomly displaced structures.

    Parameters
    ----------
    structure : Structure.
        Pymatgen structures object.
    n_struct : int.
        Total number of randomly displaced structures to be generated.
    cell_factor_sequence: list[float]
        list of factors to resize cell parameters.
    std_dev: float
        Standard deviation std_dev for normal distribution to draw numbers from to generate the rattled structures.

    Returns
    -------
    Response.output.
        Randomly displaced structures.
    """
    random_rattled = []
    if cell_factor_sequence is None:
        cell_factor_sequence = [0.975, 1.0, 1.025, 1.05]
    for cell_factor in cell_factor_sequence:
        ase_structure = AseAtomsAdaptor.get_atoms(structure)
        ase_structure.set_cell(ase_structure.get_cell() * cell_factor, scale_atoms=True)
        for seed in np.random.permutation(100000)[:n_struct]:
            ase_structure.rattle(seed=seed, stdev=std_dev)
            random_rattled.append(AseAtomsAdaptor.get_structure(ase_structure))
    return random_rattled


@job
def scale_cell(
    structure: Structure,
    scale_factor_range: list[float] | None = None,
    # n_intervals: int = 10,
):
    """
    Take in a pymatgen Structure object and generates stretched or compressed structures.

    Parameters
    ----------
    structure : Structure.
        Pymatgen structures object.
    scale_factor_range: array
        [min, max] of lattice scale factors.
    n_intervals: int.
        Number of intervals between min and max scale factors.
    scale_factors: list[float]
        List of manually specified lattice scale factors.

    Returns
    -------
    Response.output.
        Stretched or compressed structures.
    """
    atoms = AseAtomsAdaptor.get_atoms(structure)
    distorted_cells = []
    scale_factors = [0.95, 0.98, 0.99, 1.01, 1.02, 1.05]

    if scale_factor_range is None:
        # if haven't specified range, use default (or manually specified) scale_factors
        scale_factor_range = scale_factors
        print("Using default lattice scale factors of", scale_factors)
    else:
        # if have specified range
        print("Using custom lattice scale factors of", scale_factor_range)

    for i in range(len(scale_factor_range)):
        # make copy of ground state
        cell = atoms.copy()
        # set lattice parameter scale factor
        lattice_scale_factor = scale_factor_range[i]
        # scale cell volume and atomic positions
        cell.set_cell(lattice_scale_factor * atoms.get_cell(), scale_atoms=True)
        # store scaled cell
        distorted_cells.append(AseAtomsAdaptor.get_structure(cell))

    return distorted_cells


@job
def check_distances(structure: Structure, min_distance: float):
    """
    Take in a pymatgen Structure object and checks distances between atoms using minimum image convention.

    Useful after distorting cell angles and rattling to check atoms aren't too close.

    Parameters
    ----------
    structure : Structure.
        Pymatgen structures object.
    min_distance: float
        Minimum separation allowed between any two atoms.

    Returns
    -------
    Response.output.
        "True" if atoms are sufficiently spaced out i.e. all pairwise interatomic distances > min_distance.
    """
    atoms = AseAtomsAdaptor.get_atoms(structure)

    for i in range(len(atoms)):
        indices = [j for j in range(len(atoms)) if j != i]
        distances = atoms.get_distances(i, indices, mic=True)
        # print(distances)
        for distance in distances:
            if distance < min_distance:
                print("Atoms too close.")
                return False
    return True


@job
def random_vary_angle(
    structure: Structure,
    min_distance: float,
    scale: float = 10,
    wangle: list[float] | None = None,
    n_structures: int = 8,
):
    """
    Take in a pymatgen Structure object and generates angle-distorted structures.

    Parameters
    ----------
    structure : Structure.
        Pymatgen structures object.
    min_distance: float
        Minimum separation allowed between atoms.
    scale: float
        Angle scaling factor i.e. scale=10 will randomly distort angles by +-10% of original value.
    wangle: list[float]
        List of angle indices to be changed i.e. 0=alpha, 1=beta, 2=gamma.
    n_structures: int.
        Number of angle-distorted structures to generate.

    Returns
    -------
    Response.output.
        Angle-distorted structures.
    """
    atoms = AseAtomsAdaptor.get_atoms(structure)
    distorted_angle_cells = []
    generated_structures = 0  # Counter to keep track of generated structures

    if wangle is None:
        wangle = [0, 1, 2]

    while generated_structures < n_structures:
        # make copy of ground state
        atoms_copy = atoms.copy()

        # stretch lattice parameters by 3% before changing angles
        # helps atoms to not be too close
        distorted_cells = scale_cell(atoms_copy, scale_factors=[1.03])

        # getting stretched cell out of array
        newcell = distorted_cells[0].cell.cellpar()

        # current angles
        alpha = atoms_copy.cell.cellpar()[3]
        beta = atoms_copy.cell.cellpar()[4]
        gamma = atoms_copy.cell.cellpar()[5]

        # convert angle distortion scale
        scale = scale / 100
        min_scale = 1 - scale
        max_scale = 1 + scale

        while True:
            # new random angles within +-10% (default) of current angle
            new_alpha = random.randint(int(alpha * min_scale), int(alpha * max_scale))
            new_beta = random.randint(int(beta * min_scale), int(beta * max_scale))
            new_gamma = random.randint(int(gamma * min_scale), int(gamma * max_scale))

            newvalues = [new_alpha, new_beta, new_gamma]

            for wang, newv in zip(wangle, newvalues):
                newcell[wang + 3] = newv

            # converting newcell back into an Atoms object so future functions work
            # scaling atoms to new distorted cell
            atoms_copy.set_cell(newcell, scale_atoms=True)

            # if successful structure generated, i.e. atoms are not too close, then break loop
            if check_distances(atoms_copy, min_distance):
                # store scaled cell
                distorted_angle_cells.append(AseAtomsAdaptor.get_structure(atoms_copy))
                generated_structures += 1
                break  # Break the inner loop if successful

    return distorted_angle_cells


@job
def std_rattle(structure: Structure, n_structures: int = 5):
    """
    Take in a pymatgen Structure object and generates rattled structures.

    Uses standard ASE rattle.

    Parameters
    ----------
    structure : Structure.
        Pymatgen structures object.
    n_structures: int.
        Number of rattled structures to generate.

    Returns
    -------
    Response.output.
        Rattled structures.
    """
    atoms = AseAtomsAdaptor.get_atoms(structure)
    rattled_xtals = []
    seed = 42
    for i in range(n_structures):
        if i == 0:
            copy = atoms.copy()
            copy.rattle(stdev=0.01, seed=seed)
            rattled_xtals.append(copy)
        if i > 0:
            seed = seed + 1
            copy = atoms.copy()
            copy.rattle(stdev=0.01, seed=seed)
            rattled_xtals.append(AseAtomsAdaptor.get_structure(copy))
    return rattled_xtals


@job
def mc_rattle(
    structure: Structure,
    n_structures: int = 5,
    rattle_std: float = 0.003,
    min_distance: float = 1.9,
    seed: int = 42,
    n_iter: int = 10,
):
    """
    Take in a pymatgen Structure object and generates rattled structures.

    Randomly draws displacements with a MC trial step that penalises displacements leading to very small interatomic
    distances.
    Displacements generated will roughly be n_iter**0.5 * rattle_std for small values of n_iter.
    See https://hiphive.materialsmodeling.org/moduleref/structures.html?highlight=generate_mc_rattled_structures for
    more details.

    Parameters
    ----------
    structure : Structure.
        Pymatgen structures object.
    n_structures: int.
        Number of rattled structures to generate.
    rattle_std: float.
        Rattle amplitude (standard deviation in normal distribution). N.B. this value is not connected to the final
        average displacement for the structures.
    min_distance: float.
        Minimum separation of any two atoms in the rattled structures. Used for computing the probability for each
        rattle move.
    seed: int.
        Seed for setting up NumPy random state from which random numbers are generated.
    n_iter: int.
        Number of Monte Carlo iterations. Larger number of iterations will generate larger displacements.

    Returns
    -------
    Response.output.
        Monte-Carlo rattled structures.
    """
    atoms = AseAtomsAdaptor.get_atoms(structure)
    mc_rattle = generate_mc_rattled_structures(
        atoms=atoms,
        n_structures=n_structures,
        rattle_std=rattle_std,
        d_min=min_distance,
        seed=seed,
        n_iter=n_iter,
    )
    return [AseAtomsAdaptor.get_structure(xtal) for xtal in mc_rattle]
