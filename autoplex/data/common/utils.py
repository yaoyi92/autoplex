"""Utility functions for training data jobs."""
from __future__ import annotations

import random
import warnings
from typing import TYPE_CHECKING

import numpy as np
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import Trajectory as AseTrajectory
from hiphive.structure_generation import generate_mc_rattled_structures
from pymatgen.io.ase import AseAtomsAdaptor

if TYPE_CHECKING:
    from pymatgen.core import Structure


def to_ase_trajectory(
    traj_obj, filename: str = "atoms.traj", store_magmoms=False
) -> AseTrajectory:  # adapted from ase
    """
    Convert to an ASE .Trajectory.

    Parameters
    ----------
    filename : str | None
        Name of the file to write the ASE trajectory to.
        If None, no file is written.
    """
    for idx in range(len(traj_obj["atom_positions"])):
        atoms = Atoms(symbols=list(traj_obj["atomic_number"]))  # .atoms.copy()
        atoms.set_positions(traj_obj["atom_positions"][idx])
        atoms.set_cell(traj_obj["cell"][idx])
        atoms.set_pbc("T T T")

        kwargs = {
            "energy": traj_obj["energy"][idx],
            "forces": traj_obj["forces"][idx],
            "stress": traj_obj["stresses"][idx],
        }
        if store_magmoms:
            kwargs["magmom"] = traj_obj.magmoms[idx]

        atoms.calc = SinglePointCalculator(atoms=atoms, **kwargs)
        with AseTrajectory(filename, "a" if idx > 0 else "w", atoms=atoms) as f:
            f.write()

    return AseTrajectory(filename, "r")


def scale_cell(
    structure: Structure,
    volume_scale_factor_range: list[float] | None = None,
    n_structures: int = 10,
    volume_custom_scale_factors: list[float] | None = None,
):
    """
    Take in a pymatgen Structure object and generates stretched or compressed structures.

    Parameters
    ----------
    structure : Structure.
        Pymatgen structures object.
    volume_scale_factor_range : list[float]
        [min, max] of volume scale factors.
    n_structures : int.
        If specified a range, the number of structures to be generated with
        volume distortions equally spaced between min and max.
    volume_custom_scale_factors : list[float]
        Specify explicit scale factors (if range is not specified).
        If None, will default to [0.90, 0.95, 0.98, 0.99, 1.01, 1.02, 1.05, 1.10].

    Returns
    -------
    Response.output.
        Stretched or compressed structures.
    """
    atoms = AseAtomsAdaptor.get_atoms(structure)
    distorted_cells = []

    if volume_scale_factor_range is not None:
        # range is specified
        step = (
            volume_scale_factor_range[1] - volume_scale_factor_range[0]
        ) / n_structures
        scale_factors_defined = np.arange(
            volume_scale_factor_range[0], volume_scale_factor_range[1] + step, step
        )
        warnings.warn("Generated lattice scale factors within your range", stacklevel=2)

    else:  # range is not specified
        if volume_custom_scale_factors is None:
            # use default scale factors if not specified
            scale_factors_defined = [0.90, 0.95, 0.98, 0.99, 1.01, 1.02, 1.05, 1.10]
            warnings.warn(
                "Using default lattice scale factors of [0.90, 0.95, 0.98, 0.99, 1.01, 1.02, 1.05, 1.10]",
                stacklevel=2,
            )
        else:
            scale_factors_defined = volume_custom_scale_factors
            warnings.warn("Using your custom lattice scale factors", stacklevel=2)

    for i in range(len(scale_factors_defined)):
        # make copy of ground state
        cell = atoms.copy()
        # set lattice parameter scale factor
        lattice_scale_factor = (scale_factors_defined[i]) ** (1 / 3)
        # scale cell volume and atomic positions
        cell.set_cell(lattice_scale_factor * atoms.get_cell(), scale_atoms=True)
        # store scaled cell
        distorted_cells.append(AseAtomsAdaptor.get_structure(cell))

    return distorted_cells


def check_distances(structure: Structure, min_distance: float = 1.5):
    """
    Take in a pymatgen Structure object and checks distances between atoms using minimum image convention.

    Useful after distorting cell angles and rattling to check atoms aren't too close.

    Parameters
    ----------
    structure : Structure.
        Pymatgen structures object.
    min_distance: float
        Minimum separation allowed between any two atoms. Default= 1.5A.

    Returns
    -------
    Response.output.
        "True" if atoms are sufficiently spaced out i.e. all pairwise interatomic distances > min_distance.
    """
    atoms = AseAtomsAdaptor.get_atoms(structure)

    for i in range(len(atoms)):
        indices = [j for j in range(len(atoms)) if j != i]
        distances = atoms.get_distances(i, indices, mic=True)
        for distance in distances:
            if distance < min_distance:
                warnings.warn("Atoms too close.", stacklevel=2)
                return False
    return True


def random_vary_angle(
    structure: Structure,
    min_distance: float = 1.5,
    angle_percentage_scale: float = 10,
    wangle: list[float] | None = None,
    n_structures: int = 8,
    angle_max_attempts: int = 1000,
):
    """
    Take in a pymatgen Structure object and generates angle-distorted structures.

    Parameters
    ----------
    structure : Structure.
        Pymatgen structures object.
    min_distance: float
        Minimum separation allowed between atoms. Default= 1.5A.
    angle_percentage_scale: float
        Angle scaling factor.
        Default= 10 will randomly distort angles by +-10% of original value.
    wangle: list[float]
        List of angle indices to be changed i.e. 0=alpha, 1=beta, 2=gamma.
        Default= [0, 1, 2].
    n_structures: int.
        Number of angle-distorted structures to generate.
    angle_max_attempts: int.
        Maximum number of attempts to distort structure before aborting.
        Default=1000.

    Returns
    -------
    Response.output.
        Angle-distorted structures.
    """
    atoms = AseAtomsAdaptor.get_atoms(structure)
    distorted_angle_cells = []
    generated_structures = 0  # counter to keep track of generated structures

    if wangle is None:
        wangle = [0, 1, 2]

    while generated_structures < n_structures:
        attempts = 0
        # make copy of ground state
        atoms_copy = atoms.copy()

        # stretch lattice parameters by 3% before changing angles
        # helps atoms to not be too close
        distorted_cells = scale_cell(atoms_copy, volume_custom_scale_factors=[1.03])

        # getting stretched cell out of array
        newcell = distorted_cells[0].cell.cellpar()

        # current angles
        alpha = atoms_copy.cell.cellpar()[3]
        beta = atoms_copy.cell.cellpar()[4]
        gamma = atoms_copy.cell.cellpar()[5]

        # convert angle distortion scale
        angle_percentage_scale = angle_percentage_scale / 100
        min_scale = 1 - angle_percentage_scale
        max_scale = 1 + angle_percentage_scale

        while attempts < angle_max_attempts:
            attempts += 1
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
                break  # break the inner loop if successful
        else:
            raise RuntimeError(
                "Maximum attempts (1000) reached without distorting angles successfully."
            )

    return distorted_angle_cells


def std_rattle(
    structure: Structure,
    n_structures: int = 5,
    rattle_std: float = 0.01,
    rattle_seed: int = 42,
):
    """
    Take in a pymatgen Structure object and generates rattled structures.

    Uses standard ASE rattle.

    Parameters
    ----------
    structure : Structure.
        Pymatgen structures object.
    n_structures: int.
        Number of rattled structures to generate.
    rattle_std: float.
        Rattle amplitude (standard deviation in normal distribution). Default=0.01.
    rattle_seed: int.
        Seed for setting up NumPy random state from which random numbers are generated. Default= 42.

    Returns
    -------
    Response.output.
        Rattled structures.
    """
    atoms = AseAtomsAdaptor.get_atoms(structure)
    rattled_xtals = []
    for i in range(n_structures):
        if i == 0:
            copy = atoms.copy()
            copy.rattle(stdev=rattle_std, seed=rattle_seed)
            rattled_xtals.append(AseAtomsAdaptor.get_structure(copy))
        if i > 0:
            rattle_seed = rattle_seed + 1
            copy = atoms.copy()
            copy.rattle(stdev=rattle_std, seed=rattle_seed)
            rattled_xtals.append(AseAtomsAdaptor.get_structure(copy))
    return rattled_xtals


def mc_rattle(
    structure: Structure,
    n_structures: int = 5,
    rattle_std: float = 0.003,
    min_distance: float = 1.5,
    rattle_seed: int = 42,
    rattle_mc_n_iter: int = 10,
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
        average displacement for the structures. Default= 0.003.
    min_distance: float.
        Minimum separation of any two atoms in the rattled structures. Used for computing the probability for each
        rattle move. Default= 1.5A.
    rattle_seed: int.
        Seed for setting up NumPy random state from which random numbers are generated. Default= 42.
    rattle_mc_n_iter: int.
        Number of Monte Carlo iterations. Larger number of iterations will generate larger displacements. Default=10.

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
        seed=rattle_seed,
        n_iter=rattle_mc_n_iter,
    )
    return [AseAtomsAdaptor.get_structure(xtal) for xtal in mc_rattle]
