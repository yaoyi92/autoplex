"""Atomistic Jobs to Benchmark Potentials."""
from __future__ import annotations

from typing import TYPE_CHECKING

from jobflow import Response, job

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure
    from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine

from autoplex.benchmark.phonons.utils import compare_plot, get_rmse, rmse_kdep_plot


@job
def compute_bandstructure_benchmark_metrics(
    structure: Structure,
    ml_phonon_bs: PhononBandStructureSymmLine,
    dft_phonon_bs: PhononBandStructureSymmLine,
    ml_imag_modes: bool,
    dft_imag_modes: bool,
):
    """
    Compute phonon band-structure benchmark metrics and generate associated plots.

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
    file_name = f"{structure.composition.reduced_formula}_rmse_phonons.pdf"
    _ = rmse_kdep_plot(
        ml_bs=ml_phonon_bs,
        dft_bs=dft_phonon_bs,
        which_k_path=2,
        file_name=file_name,
        img_format="pdf",
    )

    # saves DFT and ML phonon band-structure overlay plot
    file_name = f"{structure.composition.reduced_formula}_band_comparison.pdf"
    _ = compare_plot(
        ml_bs=ml_phonon_bs,
        dft_bs=dft_phonon_bs,
        file_name=file_name,
    )

    return Response(
        output={
            "benchmark_phonon_rmse": overall_rmse,
            "dft_imaginary_modes": dft_imag_modes,
            "ml_imaginary_modes": ml_imag_modes,
        }
    )  # TODO TaskDoc


@job
def write_benchmark_metrics(
    ml_models: list[str],
    benchmark_structures: list[Structure],
    mp_ids: list[str],
    metrics: list,
    displacements: list[float],
    hyper_list=None,
):
    """
    Generate a text file with evaluated benchmark metrics.

    Parameters
    ----------
    benchmark_structures: List[Structure].
        Structure used for benchmarking.
    mp_ids: List[str]
        materials project ID corresponding to the structure
    metrics: List[float]
        root mean squared error between band structures
    displacements: List[float]
        Phonon displacement used for phonon computations
    hyper_list:
        List of tested atomwise regularization parameter and SOAP hyperparameters.

    Returns
    -------
    A text file with root mean squared error between DFT and ML potential phonon band-structure
    """
    print("METRICS", metrics)
    print("HYPER", hyper_list)
    if hyper_list is None:
        hyper_list = ["default"]
    metrics_flattened = [item for sublist in metrics for item in sublist]
    for ml_model in ml_models:
        for benchmark_structure, mp_id in zip(benchmark_structures, mp_ids):
            structure_composition = benchmark_structure.composition.reduced_formula
            with open(
                f"results_{structure_composition}.txt",
                "a",
                encoding="utf-8",
            ) as file:
                file.write(
                    "%-11s%-11s%-12s%-14s%-6s%-55s%-16s%-16s"
                    % (
                        "Potential",
                        "Structure",
                        "MPID",
                        "Displacement",
                        "RMSE",
                        "Hyperparameters (atom-wise f, n_sparse, SOAP delta)",
                        "imagmodes(pot)",
                        "imagmodes(dft)",
                    )
                )
            for displacement in displacements:
                for metric, hyper in zip(metrics_flattened, hyper_list):
                    with open(
                        f"results_{structure_composition}.txt",
                        "a",
                        encoding="utf-8",
                    ) as file:
                        file.write(
                            "\n%-11s%-11s%-12s%-14.2f%-6.3f%-55s%-16s%-16s"
                            % (
                                ml_model,
                                structure_composition,
                                mp_id,
                                displacement,
                                metric["benchmark_phonon_rmse"],
                                str(hyper),
                                str(metric["ml_imaginary_modes"]),
                                str(metric["dft_imaginary_modes"]),
                            )
                        )

    return Response(output=metrics)
