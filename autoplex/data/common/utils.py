"""Utility functions for training data jobs."""
from __future__ import annotations

# ase imports
import ase.io
import matplotlib.pyplot as plt

# general imports
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import Trajectory as AseTrajectory
from ase.io import write


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


def rms_dict(x_ref, x_pred):
    """
    Take two datasets of the same shape and returns a dictionary containing RMS error data.

    Parameters
    ----------
    x_ref:
        Reference data
    x_pred:
        Predicted data

    """
    x_ref = np.array(x_ref)
    x_pred = np.array(x_pred)

    if np.shape(x_pred) != np.shape(x_ref):
        raise ValueError("WARNING: not matching shapes in rms")

    error_2 = (x_ref - x_pred) ** 2

    average = np.sqrt(np.average(error_2))
    std_ = np.sqrt(np.var(error_2))

    return {"rmse": average, "std": std_}


def sort_outlier_energy(in_file, out_file, criteria: float = 0.0005):
    """
    Sort data outliers per energy criteria and write them into files.

    Parameters
    ----------
    in_file:
        Reference file (e.g. DFT).
    out_file:
        MLIP generated data file.

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

    write("sorted_in_energy.extxyz", atoms_in, append=True)
    write("sorted_out_energy.extxyz", atoms_out, append=True)
    write("outliers_energy.extxyz", outliers, append=True)


def sort_outlier_forces(in_file, out_file, symbol="Si", criteria: float = 0.1):
    """
    Sort data outliers per force criteria and write them into files.

    Parameters
    ----------
    in_file:
        Reference file (e.g. DFT).
    out_file:
        MLIP generated data file.

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
        if not any(np.any(array > criteria) for array in force_error):
            atoms_in.append(at_in)
            atoms_out.append(at_out)
        else:
            outliers.append(at_in)

    write("sorted_in_force.extxyz", atoms_in, append=True)
    write("sorted_out_force.extxyz", atoms_out, append=True)
    write("outliers_force.extxyz", outliers, append=True)


# copied from libatoms GAP tutorial page and adapted
def energy_plot(in_file, out_file, ax, title="Plot of energy"):
    """
    Plot the distribution of energy per atom on the output vs the input.

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

    """
    # read files
    in_atoms = ase.io.read(in_file, ":")
    for atoms in in_atoms:
        kwargs = {
            "energy": atoms.info["REF_energy"],
        }
        atoms.calc = SinglePointCalculator(atoms=atoms, **kwargs)
    out_atoms = ase.io.read(out_file, ":")
    # list energies
    ener_in = [
        at.get_potential_energy() / len(at.get_chemical_symbols()) for at in in_atoms
    ]
    ener_out = [
        at.get_potential_energy() / len(at.get_chemical_symbols()) for at in out_atoms
    ]
    # scatter plot of the data
    ax.scatter(ener_in, ener_out)
    # get the appropriate limits for the plot
    for_limits = np.array(ener_in + ener_out)
    elim = (for_limits.min(), for_limits.max())
    ax.set_xlim(elim)
    ax.set_ylim(elim)
    # add line of slope 1 for reference
    ax.plot(elim, elim, c="k")
    # set labels
    ax.set_ylabel("energy by GAP / eV")
    ax.set_xlabel("energy by DFT / eV")
    # set title
    ax.set_title(title)
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
    in_file, out_file, ax, symbol="Si", title="Plot of force"
):  # make general symbol
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
    ax.scatter(in_force, out_force)
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


def plot_energy_forces(title: str, energy_limit: float, force_limit: float):
    """
    Plot energy and forces of the data.

    Parameters
    ----------
    title:
        Title of the plot.


    """
    fig, ax_list = plt.subplots(nrows=3, ncols=2, gridspec_kw={"hspace": 0.3})
    fig.set_size_inches(15, 20)
    ax_list = ax_list.flat[:]

    sort_outlier_energy("train.extxyz", "quip_train.extxyz", energy_limit)
    sort_outlier_energy("test.extxyz", "quip_test.extxyz", energy_limit)
    sort_outlier_forces("train.extxyz", "quip_train.extxyz", "Si", force_limit)
    sort_outlier_forces("test.extxyz", "quip_test.extxyz", "Si", force_limit)

    energy_plot(
        "train.extxyz", "quip_train.extxyz", ax_list[0], "Energy on training data"
    )
    force_plot(
        "train.extxyz",
        "quip_train.extxyz",
        ax_list[1],
        "Si",
        "Force on training data - Si",
    )
    energy_plot("test.extxyz", "quip_test.extxyz", ax_list[2], "Energy on test data")
    force_plot(
        "test.extxyz", "quip_test.extxyz", ax_list[3], "Si", "Force on test data - Si"
    )
    energy_plot(
        "sorted_in_energy.extxyz",
        "sorted_out_energy.extxyz",
        ax_list[4],
        "Energy on sorted data",
    )
    force_plot(
        "sorted_in_energy.extxyz",
        "sorted_out_energy.extxyz",
        ax_list[5],
        "Si",
        "Force on sorted data - Si",
    )
    energy_plot(
        "sorted_in_force.extxyz",
        "sorted_out_force.extxyz",
        ax_list[4],
        "Energy on sorted data",
    )
    force_plot(
        "sorted_in_force.extxyz",
        "sorted_out_force.extxyz",
        ax_list[5],
        "Si",
        "Force on sorted data - Si",
    )

    fig.suptitle(title, fontsize=16)

    plt.savefig("energy_forces.eps", format="eps")
    plt.savefig("energy_forces.png")
    plt.show()
