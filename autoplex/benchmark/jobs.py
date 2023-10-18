"""
Atomistic Jobs to Benchmark Potentials
"""
from __future__ import annotations

from jobflow import Response, job
from pymatgen.core.structure import Structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from autoplex.benchmark.utils import get_rmse, rmse_kdep_plot, compare_plot


@job
def compute_bandstructure_benchmark_metrics(
    structure: Structure,
    ml_phonon_bs: PhononBandStructureSymmLine,
    dft_phonon_bs: PhononBandStructureSymmLine,
):  # TODO include which pot. method has been used (GAP, ACE, etc.)
    """
    Computes phonon band-structure benchmark metrics and associated plots.

    Parameters
    ----------
    structure : .Structure
     A structure object.
    ml_phonon_bs: PhononBandStructureSymmLine.
       ML generated pymatgen phonon band-structure object
    dft_phonon_bs: PhononBandStructureSymmLine.
       DFT generated pymatgen phonon band-structure object

    Returns
    -------
    Response.output
       Overall root mean squared error between DFT and ML phonon band-structure.
    """
    # compute overall root mean squared error
    overall_rmse = get_rmse(ml_bs=ml_phonon_bs, dft_bs=dft_phonon_bs)

    # saves rmse k-dependent plot
    file_name = f"{structure.composition.reduced_formula}_rmse_phonons.eps"
    _ = rmse_kdep_plot(
        ml_bs=ml_phonon_bs,
        dft_bs=dft_phonon_bs,
        which_k_path=2,
        file_name=file_name,
        img_format="eps",
    )

    # saves DFT and ML phonon band-structure overlay plot
    file_name = f"{structure.composition.reduced_formula}_band_comparison.eps"
    _ = compare_plot(
        ml_bs=ml_phonon_bs,
        dft_bs=dft_phonon_bs,
        file_name=file_name,
        img_format="eps",
    )

    return Response(output=overall_rmse)  # TODO TaskDoc
