"""Utility functions for training data jobs."""
from __future__ import annotations

import random
import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pymatgen.core import Structure

import ase.io
import matplotlib.pyplot as plt
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import Trajectory as AseTrajectory
from ase.io import write
from hiphive.structure_generation import generate_mc_rattled_structures
from pymatgen.io.ase import AseAtomsAdaptor


def rms_dict(x_ref, x_pred) -> dict:
    """Compute RMSE and standard deviation of predictions with reference data.

    x_ref and x_pred should be of same shape.

    Parameters
    ----------
    x_ref : np.ndarray.
        list of reference data.
    x_pred: np.ndarray.
        list of prediction.

    Returns
    -------
    dict
        Dict with RMSE and std deviation of predictions.
    """
    x_ref = np.array(x_ref)
    x_pred = np.array(x_pred)
    if np.shape(x_pred) != np.shape(x_ref):
        raise ValueError("WARNING: not matching shapes in rms")
    error_2 = (x_ref - x_pred) ** 2
    average = np.sqrt(np.average(error_2))
    std_ = np.sqrt(np.var(error_2))
    return {"rmse": average, "std": std_}


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
) -> list[Structure]:
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
        scale_factors_defined = np.arange(
            volume_scale_factor_range[0],
            volume_scale_factor_range[1]
            + (volume_scale_factor_range[1] - volume_scale_factor_range[0])
            / (n_structures - 1),
            (volume_scale_factor_range[1] - volume_scale_factor_range[0])
            / (n_structures - 1),
        )

        if 1 not in scale_factors_defined:
            scale_factors_defined = np.append(scale_factors_defined, 1)
            scale_factors_defined = np.sort(scale_factors_defined)

        warnings.warn(
            f"Generated lattice scale factors {scale_factors_defined} within your range",
            stacklevel=2,
        )

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

    for scale_factor in scale_factors_defined:
        # make copy of ground state
        cell = atoms.copy()
        # set lattice parameter scale factor
        lattice_scale_factor = scale_factor ** (1 / 3)
        # scale cell volume and atomic positions
        cell.set_cell(lattice_scale_factor * atoms.get_cell(), scale_atoms=True)
        # store scaled cell
        distorted_cells.append(AseAtomsAdaptor.get_structure(cell))
    return distorted_cells


def check_distances(structure: Structure, min_distance: float = 1.5) -> bool:
    """
    Take in a pymatgen Structure object and check minimum distances between atoms using minimum image convention.

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
    w_angle: list[float] | None = None,
    n_structures: int = 8,
    angle_max_attempts: int = 1000,
) -> list[Structure]:
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
    w_angle: list[float]
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

    if w_angle is None:
        w_angle = [0, 1, 2]

    while generated_structures < n_structures:
        attempts = 0
        # make copy of ground state
        atoms_copy = atoms.copy()

        # stretch lattice parameters by 3% before changing angles
        # helps atoms to not be too close
        distorted_cells = scale_cell(
            AseAtomsAdaptor.get_structure(atoms_copy),
            volume_custom_scale_factors=[1.03],
        )

        distorted_supercells: Atoms = AseAtomsAdaptor.get_atoms(distorted_cells[0])

        # getting stretched supercell out of array
        newcell = distorted_supercells.cell.cellpar()

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

            for wang, newv in zip(w_angle, newvalues):
                newcell[wang + 3] = newv

            # converting newcell back into an Atoms object so future functions work
            # scaling atoms to new distorted cell
            atoms_copy.set_cell(newcell, scale_atoms=True)

            # if successful structure generated, i.e. atoms are not too close, then break loop
            if check_distances(AseAtomsAdaptor.get_structure(atoms_copy), min_distance):
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
) -> list[Structure]:
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
) -> list[Structure]:
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


def extract_base_name(filename, is_out=False) -> str:
    """
    Extract the base of a file name to easier manipulate other file names.

    Parameters
    ----------
    filename:
        The name of the file.
    is_out: bool
        If it is an out_file (i.e. prefix is "quip_")

    """
    if is_out:
        # Extract "quip_train" or "quip_test"
        if "quip_train" in filename:
            return "quip_train"
        if "quip_test" in filename:
            return "quip_test"
    else:
        # Extract "train" or "test"
        base_name = filename.split(".", 1)[0]
        return base_name.split("_", 1)[0]

    return "A problem with the files occurred."


def filter_outlier_energy(in_file, out_file, criteria: float = 0.0005) -> None:
    """
    Filter data outliers per energy criteria and write them into files.

    Parameters
    ----------
    in_file:
        Reference file (e.g. DFT).
    out_file:
        MLIP generated data file.
    criteria:
        Energy filter threshold.

    """
    # read files
    in_atoms = ase.io.read(in_file, ":")
    for atoms in in_atoms:
        kwargs = {
            "energy": atoms.info["REF_energy"],
        }
        atoms.calc = SinglePointCalculator(atoms=atoms, **kwargs)
    out_atoms = ase.io.read(out_file, ":")

    atoms_in = []
    atoms_out = []
    outliers = []

    for at_in, at_out in zip(in_atoms[:-1], out_atoms):
        en_in = at_in.get_potential_energy() / len(at_in.get_chemical_symbols())
        en_out = at_out.get_potential_energy() / len(at_out.get_chemical_symbols())
        en_error = rms_dict(en_in, en_out)
        if abs(en_error["rmse"]) < criteria:
            atoms_in.append(at_in)
            atoms_out.append(at_out)
        else:
            outliers.append(at_in)

    write(
        in_file.replace(extract_base_name(in_file), "filtered_in_energy"),
        atoms_in,
        append=True,
    )
    write(
        out_file.replace(
            extract_base_name(out_file, is_out=True), "filtered_out_energy"
        ),
        atoms_out,
        append=True,
    )
    write(
        in_file.replace(extract_base_name(in_file), "outliers_energy"),
        outliers,
        append=True,
    )


def filter_outlier_forces(
    in_file, out_file, symbol="Si", criteria: float = 0.1
) -> None:
    """
    Filter data outliers per force criteria and write them into files.

    Parameters
    ----------
    in_file:
        Reference file (e.g. DFT).
    out_file:
        MLIP generated data file.
    symbol:
        Atomi symbol.
    criteria:
        Force filter threshold.

    """
    # read files
    in_atoms = ase.io.read(in_file, ":")
    for atoms in in_atoms[:-1]:
        kwargs = {
            "forces": atoms.arrays["REF_forces"],
        }
        atoms.calc = SinglePointCalculator(atoms=atoms, **kwargs)
    out_atoms = ase.io.read(out_file, ":")
    atoms_in = []
    atoms_out = []
    outliers = []
    # extract data for only one species
    for at_in, at_out in zip(in_atoms[:-1], out_atoms):
        # get the symbols
        sym_all = at_in.get_chemical_symbols()
        # add force for each atom
        force_error = []
        for j, sym in enumerate(sym_all):
            if sym in symbol:
                in_force = at_in.get_forces()[j]
                out_force = at_out.arrays["force"][j]
                rms = rms_dict(in_force, out_force)
                force_error.append(rms["rmse"])
        at_in.info["max RMSE"] = max(force_error) if force_error else 0
        at_in.info["avg RMSE"] = (
            sum(force_error) / len(force_error) if force_error else 0
        )
        at_in.info["RMSE"] = force_error
        if not any(np.any(value > criteria) for value in force_error):
            atoms_in.append(at_in)
            atoms_out.append(at_out)
        else:
            outliers.append(at_in)

    write(
        in_file.replace(extract_base_name(in_file), "filtered_in_force"),
        atoms_in,
        append=True,
    )
    write(
        out_file.replace(
            extract_base_name(out_file, is_out=True), "filtered_out_force"
        ),
        atoms_out,
        append=True,
    )
    write(
        in_file.replace(extract_base_name(in_file), "outliers_force"),
        outliers,
        append=True,
    )


def energy_plot(
    in_file, out_file, ax, title: str = "Plot of energy", label: str = "energy"
) -> None:
    """
    Plot the distribution of energy per atom on the output vs the input.

    Adapted and adjusted from libatoms GAP tutorial page https://libatoms.github.io/GAP/gap_fitting_tutorial.html.

    Parameters
    ----------
    in_file:
        Reference file (e.g. DFT).
    out_file:
        MLIP generated data file.
    ax:
        Panel position in plt.subplots.
    title:
        Title of the plot.
    label:
        Legend label

    """
    # read files
    in_atoms = ase.io.read(in_file, ":")
    out_atoms = ase.io.read(out_file, ":")
    atoms_in: list = []
    atoms_out: list = []
    for at_in, at_out in zip(in_atoms, out_atoms):
        kwargs = {
            "energy": at_in.info["REF_energy"],
        }
        at_in.calc = SinglePointCalculator(atoms=at_in, **kwargs)
        for at, atoms in [(at_in, atoms_in), (at_out, atoms_out)]:
            if at.info["config_type"] not in {"isolated_atom", "IsolatedAtom"}:
                atoms.append(at)
    # list energies
    ener_in = [
        at.get_potential_energy() / len(at.get_chemical_symbols()) for at in atoms_in
    ]
    ener_out = [
        at.get_potential_energy() / len(at.get_chemical_symbols()) for at in atoms_out
    ]
    # scatter plot of the data
    ax.scatter(ener_in, ener_out, label=label)
    # get the appropriate limits for the plot
    for_limits = np.array(ener_in + ener_out)
    elim = (for_limits.min() - 0.01, for_limits.max() + 0.01)
    ax.set_xlim(elim)
    ax.set_ylim(elim)
    # add line of slope 1 for reference
    ax.plot(elim, elim, c="k")
    # set labels
    ax.set_ylabel("energy by GAP / eV")
    ax.set_xlabel("energy by DFT / eV")
    # set title
    ax.set_title(title)
    # set legend
    ax.legend(loc="upper right")
    # add text about RMSE
    _rms = rms_dict(ener_in, ener_out)
    rmse_text = (
        "RMSE:\n"
        + str(np.round(_rms["rmse"], 3))
        + " +- "
        + str(np.round(_rms["std"], 3))
        + "eV/atom"
    )
    ax.text(
        0.9,
        0.1,
        rmse_text,
        transform=ax.transAxes,
        fontsize="large",
        horizontalalignment="right",
        verticalalignment="bottom",
    )


def force_plot(
    in_file,
    out_file,
    ax,
    symbol: str = "Si",
    title: str = "Plot of force",
    label: str = "force for ",
) -> float:
    """
    Plot the distribution of force components per atom on the output vs the input.

    Only plots for the given atom type(s).

    Parameters
    ----------
    in_file:
        Reference file (e.g. DFT).
    out_file:
        MLIP generated data file.
    ax:
        Panel position in plt.subplots.
    symbol:
        Chemical symbol.
    title:
        Title of the plot.
    label:
        Legend label

    """
    in_atoms = ase.io.read(in_file, ":")
    for atoms in in_atoms[:-1]:
        kwargs = {
            "forces": atoms.arrays["REF_forces"],
        }
        atoms.calc = SinglePointCalculator(atoms=atoms, **kwargs)
    out_atoms = ase.io.read(out_file, ":")
    # extract data for only one species
    in_force, out_force = [], []
    for at_in, at_out in zip(in_atoms[:-1], out_atoms):
        # get the symbols
        sym_all = at_in.get_chemical_symbols()
        # add force for each atom
        for j, sym in enumerate(sym_all):
            if sym in symbol:
                in_force.append(at_in.get_forces()[j])
                # out_force.append(at_out.get_forces()[j]) \
                out_force.append(
                    at_out.arrays["force"][j]
                )  # because QUIP and ASE use different names
    # convert to np arrays, much easier to work with
    # in_force = np.array(in_force)
    # out_force = np.array(out_force)
    # scatter plot of the data
    ax.scatter(in_force, out_force, label=label + symbol)
    # get the appropriate limits for the plot
    for_limits = np.array(in_force + out_force)
    flim = (for_limits.min() - 1, for_limits.max() + 1)
    ax.set_xlim(flim)
    ax.set_ylim(flim)
    # add line of
    ax.plot(flim, flim, c="k")
    # set labels
    ax.set_ylabel("force by GAP / (eV/Å)")
    ax.set_xlabel("force by DFT / (eV/Å)")
    # set title
    ax.set_title(title)
    # set legend
    ax.legend(loc="upper right")
    # add text about RMSE
    _rms = rms_dict(in_force, out_force)
    rmse_text = (
        "RMSE:\n"
        + str(np.round(_rms["rmse"], 3))
        + " +- "
        + str(np.round(_rms["std"], 3))
        + "eV/Å"
    )
    ax.text(
        0.9,
        0.1,
        rmse_text,
        transform=ax.transAxes,
        fontsize="large",
        horizontalalignment="right",
        verticalalignment="bottom",
    )

    return _rms["rmse"]


def plot_energy_forces(
    title: str,
    energy_limit: float,
    force_limit: float,
    species_list: list | None = None,
    train_name: str = "train.extxyz",
    test_name: str = "test.extxyz",
) -> None:
    """
    Plot energy and forces of the data.

    Parameters
    ----------
    title:
        Title of the plot.
    energy_limit:
        Energy limit for data filtering.
    force_limit:
        Force limit for data filtering.
    species_list:
        List of species.
    train_name:
        name of the training data file.
    test_name:
        name of the test data file.


    """
    if species_list is None:
        species_list = ["Si"]
    fig, ax_list = plt.subplots(nrows=3, ncols=2, gridspec_kw={"hspace": 0.3})
    fig.set_size_inches(15, 20)
    ax_list = ax_list.flat[:]

    pretty_species_list = (
        str(species_list).replace("['", "").replace("']", "").replace("'", "")
    )

    energy_plot(train_name, "quip_" + train_name, ax_list[0], "Energy on training data")
    # rmse_train =
    for species in species_list:
        force_plot(
            train_name,
            "quip_" + train_name,
            ax_list[1],
            species,
            f"Force on training data - {pretty_species_list}",
        )
    energy_plot(test_name, "quip_" + test_name, ax_list[2], "Energy on test data")
    filter_outlier_energy(train_name, "quip_" + train_name, energy_limit)
    filter_outlier_energy(test_name, "quip_" + test_name, energy_limit)
    # rmse_test =
    for species in species_list:
        force_plot(
            test_name,
            "quip_" + test_name,
            ax_list[3],
            species,
            f"Force on test data - {pretty_species_list}",
        )
        filter_outlier_forces(train_name, "quip_" + train_name, species, force_limit)
        filter_outlier_forces(test_name, "quip_" + test_name, species, force_limit)

    energy_plot(
        train_name.replace("train", "filtered_in_energy"),
        train_name.replace("train", "filtered_out_energy"),
        ax_list[4],
        "Energy on filtered data",
        "energy-filtered data, energy",
    )
    for species in species_list:
        force_plot(
            train_name.replace("train", "filtered_in_force"),
            train_name.replace("train", "filtered_out_force"),
            ax_list[5],
            species,
            f"Force on filtered data - {pretty_species_list}",
            "energy-filtered data, force for ",
        )
    energy_plot(
        train_name.replace("train", "filtered_in_energy"),
        train_name.replace("train", "filtered_out_energy"),
        ax_list[4],
        "Energy on filtered data",
        "force-filtered data, energy",
    )
    for species in species_list:
        force_plot(
            train_name.replace("train", "filtered_in_force"),
            train_name.replace("train", "filtered_out_force"),
            ax_list[5],
            species,
            f"Force on filtered data - {pretty_species_list}",
            "force-filtered data, force for ",
        )

    fig.suptitle(title, fontsize=16)

    plt.savefig(
        train_name.replace("train", "energy_forces").replace(".extxyz", ".pdf"),
        format="pdf",
    )
    plt.savefig(train_name.replace("train", "energy_forces").replace(".extxyz", ".png"))


def generate_supercell_matrix(structure, supercell_matrix, max_sites=400):
    """
    Generate the updated supercell matrix.

    Parameters
    ----------
    structure: Structure.
        Pymatgen structures object.
    supercell_matrix: Matrix3D.
        Matrix for obtaining the supercell.
    max_sites: int
        maximum number of sites.

    Returns
    -------
    Updated supercell matrix

    """
    if supercell_matrix is None:
        supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

    supercell_check = structure * supercell_matrix

    if supercell_check.num_sites > max_sites:
        # estimate a factor based on the original structure and max_sites
        initial_factor = int(round(pow(max_sites / structure.num_sites, 1 / 3), 0))

        # First, try the matrices with the same factor in all dimensions
        for factor in range(initial_factor, 1, -1):
            matrix = [[factor, 0, 0], [0, factor, 0], [0, 0, factor]]
            supercell_check = structure * matrix
            if supercell_check.num_sites <= max_sites:
                return matrix

        # Then, try the matrices with the same factor in two dimensions and 1 in the third
        for factor in range(initial_factor, 1, -1):
            matrices = [
                [[factor, 0, 0], [0, factor, 0], [0, 0, 1]],
                [[factor, 0, 0], [0, 1, 0], [0, 0, factor]],
                [[1, 0, 0], [0, factor, 0], [0, 0, factor]],
            ]
            for matrix in matrices:
                supercell_check = structure * matrix
                if supercell_check.num_sites <= max_sites:
                    return matrix

        # Finally, try the matrices with the factor in one dimension and 1 in the other two
        for factor in range(initial_factor, 1, -1):
            matrices = [
                [[factor, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[1, 0, 0], [0, factor, 0], [0, 0, 1]],
                [[1, 0, 0], [0, 1, 0], [0, 0, factor]],
            ]
            for matrix in matrices:
                supercell_check = structure * matrix
                if supercell_check.num_sites <= max_sites:
                    return matrix

        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    return supercell_matrix
