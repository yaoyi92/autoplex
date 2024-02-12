"""Utility functions for benchmarking jobs."""

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.plotter import PhononBSPlotter


def get_rmse(
    ml_bs: PhononBandStructureSymmLine,
    dft_bs: PhononBandStructureSymmLine,
    k_dependent_rmse: bool = False,
):
    """
    Compute root mean squared error (rmse) between DFT and ML phonon band-structure.

    Parameters
    ----------
    ml_bs : PhononBandStructureSymmLine.
        ML generated pymatgen phonon band-structure object
    dft_bs : PhononBandStructureSymmLine.
        DFT generated pymatgen phonon band-structure object
    k_dependent_rmse : bool.
        If true, method returns k-dependent rmse between the band-structures

    Returns
    -------
    float or list[float]
      root mean squared error between DFT and ML phonon band-structure
    """
    diff = ml_bs.bands - dft_bs.bands
    rmse = np.sqrt(np.mean(diff**2))

    if k_dependent_rmse:
        diff_here = np.transpose(diff)
        rmse = [np.sqrt(np.mean(diff_here[i] ** 2)) for i in range(len(diff_here))]

    return rmse


def rmse_kdep_plot(
    ml_bs: PhononBandStructureSymmLine,
    dft_bs: PhononBandStructureSymmLine,
    which_k_path=1,
    file_name="rms.eps",
    img_format="eps",
):
    """
    Save k dependent root mean squared error plot between DFT and ML phonon band-structure.

    Parameters
    ----------
    ml_bs : PhononBandStructureSymmLine.
        ML generated pymatgen phonon band-structure object
    dft_bs : PhononBandStructureSymmLine.
        DFT generated pymatgen phonon band-structure object
    which_k_path : int.
        If set 1, use ML band-structure as reference, if 2, uses DFT band-structure as reference
    file_name: str.
        Name of the saved plot
    img_format: str
        File extension of plot to be saved, default is eps

    Returns
    -------
    matplotlib.plt
        A matplotlib figure with k-dependent rmse
    """
    rmse_kp = get_rmse(ml_bs=ml_bs, dft_bs=dft_bs, k_dependent_rmse=True)
    if which_k_path == 1:
        plotter = PhononBSPlotter(bs=ml_bs)
    elif which_k_path == 2:
        plotter = PhononBSPlotter(bs=dft_bs)

    distances = []
    for element in plotter.bs_plot_data()["distances"]:
        distances.extend(element)

    plt.close("all")
    plt.plot(distances, rmse_kp)
    plt.xticks(
        ticks=plotter.bs_plot_data()["ticks"]["distance"],
        labels=plotter.bs_plot_data()["ticks"]["label"],
    )
    plt.xlabel("Wave vector")
    plt.ylabel("Phonons RMS (THz)")
    plt.savefig(file_name, format=img_format)

    return plt


def compare_plot(
    ml_bs: PhononBandStructureSymmLine,
    dft_bs: PhononBandStructureSymmLine,
    file_name: str = "band_comparison.eps",
):
    """
    Save DFT and ML phonon band-structure overlay plot for visual comparison.

    Parameters
    ----------
    ml_bs : PhononBandStructureSymmLine.
        ML generated pymatgen phonon band-structure object
    dft_bs : PhononBandStructureSymmLine.
        DFT generated pymatgen phonon band-structure object
    file_name: str.
        Name of the saved plot

    Returns
    -------
    matplotlib.plt
        A matplotlib figure with DFT and ML generated phonon band-structure overlay
    """
    plotter = PhononBSPlotter(bs=ml_bs)
    plotter2 = PhononBSPlotter(bs=dft_bs)
    new_plotter = plotter.plot_compare(plotter2)
    new_plotter.figure.savefig(file_name)

    return new_plotter.figure
