"""
Atomistic Jobs to Benchmark Potentials
"""
from __future__ import annotations

import os
import numpy as np
from atomate2.common.schemas.phonons import PhononBSDOSDoc
from jobflow import Response, job
from pymatgen.core.structure import Structure
from pymatgen.phonon.plotter import PhononBSPlotter


@job
def compute_bandstructure_benchmark_metrics(
    structure: Structure,
    ml_phonon_task_doc: PhononBSDOSDoc,
    dft_phonon_task_doc: PhononBSDOSDoc,
):  # TODO include which pot. method has been used (GAP, ACE, etc.)
    """
    Computes phonon band-structure benchmark metrics.

    Parameters
    ----------
    structure : .Structure
     A structure object.
    ml_phonon_task_doc: PhononBSDOSDoc.
       Phonon task doc from ML potential consisting of pymatgen band-structure object
    dft_phonon_task_doc: PhononBSDOSDoc.
       Phonon task doc from DFT runs consisting of pymatgen band-structure object

    Returns
    -------
    Response.output
       Overall root mean squared error between DFT and ML phonon band-structure.
    """

    def rms_overall():
        """
        Computes overall root mean squared error between DFT and ML phonon band-structure

        Returns
        -------
        float
        """
        diff = np.array(ml_bs.as_dict()["bands"]) - np.array(dft_bs.as_dict()["bands"])
        return np.sqrt(np.mean(diff**2))

    def rms_kdep():
        """
        Computes k dependent root mean squared error between DFT and ML phonon band-structure

        Returns
        -------
        float
        """
        diff = np.array(ml_bs.as_dict()["bands"]) - np.array(dft_bs.as_dict()["bands"])

        diff = np.transpose(diff)
        kpointdep = [np.sqrt(np.mean(diff[i] ** 2)) for i in range(len(diff))]
        return kpointdep

    def rms_kdep_plot(whichkpath=1, filename="rms.eps", img_format="eps"):
        """
        Saves k dependent root mean squared error plot between DFT and ML phonon band-structure
        """
        rmskp = rms_kdep()
        if whichkpath == 1:
            plotter = PhononBSPlotter(bs=ml_bs)
        elif whichkpath == 2:
            plotter = PhononBSPlotter(bs=dft_bs)

        distances = []
        for element in plotter.bs_plot_data()["distances"]:
            distances.extend(element)

        import matplotlib.pyplot as plt  # pylint: disable=C0415

        plt.close("all")
        plt.plot(distances, rmskp)
        plt.xticks(
            ticks=plotter.bs_plot_data()["ticks"]["distance"],
            labels=plotter.bs_plot_data()["ticks"]["label"],
        )
        plt.xlabel("Wave vector")
        plt.ylabel("Phonons RMS (THz)")
        plt.savefig(filename, format=img_format)

    def compare_plot(
        filename="band_comparison.eps", img_format="eps"
    ):  # pylint: disable=W0612
        """
        Saves DFT and ML phonon band-structure overlay plot for visual comparison
        """
        plotter = PhononBSPlotter(bs=ml_bs)
        plotter2 = PhononBSPlotter(bs=dft_bs)
        new_plotter = plotter.plot_compare(plotter2)
        new_plotter.savefig(filename, format=img_format)
        new_plotter.close()

    ml_bs = ml_phonon_task_doc.phonon_bandstructure
    dft_bs = dft_phonon_task_doc.phonon_bandstructure

    rms = rms_overall()

    rms_kdep_plot(
        whichkpath=2,
        filename=os.path.join(str(structure.composition.reduced_formula))
        + "_rms_phonons.eps",
    )

    return Response(output=rms)  # TODO TaskDoc
