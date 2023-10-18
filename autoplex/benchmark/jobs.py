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


@job
def write_benchmark_metrics(benchmark_structure: Structure, mp_id, rmse, displacements):
    """
    Generate a text file with evaluated benchmark metrics

    Parameters
    ----------
    benchmark_structure: Structure.
        Structure used for benchmarking.
    mp_id: str
        materials project ID corresponding to the structure
    rmse: List[float]
        root mean squared error between band structures
    displacements: List[float]
        Phonon displacement used for phonon computations

    Returns
    -------
    A text file with root mean squared error between DFT and ML potential phonon band-structure
    """
    structure_composition = benchmark_structure.composition.reduced_formula
    with open(
        f"results_{structure_composition}.txt",
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            f"Pot Structure mpid displacements RMS imagmodes(pot) imagmodes(dft) "
            f"\nGAP {structure_composition} {mp_id} {displacements} {rmse} "
        )
        # TODO include which pot. method has been used (GAP, ACE, etc.)
        # TODO has img modes + ' ' + ' ' + str(ml.has_imag_modes(0.1))
        #  + ' ' + str(dft.has_imag_modes(0.1))

    return Response
