"""Atomistic Jobs to Benchmark Potentials."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jobflow import Response, job

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure


@job
def write_benchmark_metrics(
    benchmark_structures: list,
    metrics: list,
):
    """
    Generate a text file with evaluated benchmark metrics.

    Parameters
    ----------
    ml_models: list[str]
        list of the ML models to be used. Default is GAP.
    benchmark_structures: List[Structure].
        list of benchmark Structure used for benchmarking.
    benchmark_mp_ids: List[str]
        list of benchmark structure materials project ID.
    metrics: List[float]
        root mean squared error between band structures, imagmodesdft-bool and imagmodesml-bool.
    displacements: List[float]
        Phonon displacements used for phonon computations
    hyper_list:
        List of tested atomwise regularization parameter and SOAP hyperparameters.

    Returns
    -------
    A text file with root mean squared error between DFT and ML potential phonon band-structure
    """
    #if hyper_list is None:
    #    hyper_list = ["default"]
    # TODO: fix this part
    print(metrics)
    metrics_flattened = [item for sublist in metrics for item in sublist]
    # TODO: think about a better solution here
    # the following code assumes all benchmark structures have the same composition
    structure_composition=benchmark_structures[0].composition.reduced_formula
    with open(
            f"results_{structure_composition}.txt",
            "a",
            encoding="utf-8",
    ) as file:
        file.write(
            "%-11s%-11s%-12s%-18s%-12s%-55s%-16s%-14s"
            % (
                "Potential",
                "Structure",
                "MPID",
                "Displacement (Å)",
                "RMSE (THz)",
                "Hyperparameters (atom-wise f, n_sparse, SOAP delta)",
                "imagmodes(pot)",
                "imagmodes(dft)",
            )
        )

    for metric in metrics_flattened:
       with open(
            f"results_{structure_composition}.txt",
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                "\n%-11s%-11s%-12s%-18.2f%-12.5f%-55s%-16s%-5s"
                % (
                    metric["ml_model"],
                    structure_composition,
                    metric["mp_id"],
                    metric["displacement"],
                    metric["benchmark_phonon_rmse"],
                    str({"f=" + str(metric["atomwise_regularization_parameter"]): metric["soap_dict"]}) if metric["soap_dict"] is not None else str({"f=" + str(metric["atomwise_regularization_parameter"]): metric["suffix"]}),
                    str(metric["ml_imaginary_modes"]),
                    str(metric["dft_imaginary_modes"]),
                )
            )

    return Response(output=metrics)
