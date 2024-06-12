import json
import pytest
from pymatgen.core.structure import Structure
from pymatgen.io.phonopy import get_ph_bs_symm_line
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from autoplex.benchmark.phonons.jobs import (
    compute_bandstructure_benchmark_metrics,
    write_benchmark_metrics,
)


def test_compute_bandstructure_benchmark_metrics_dummy(test_dir, clean_dir):
    import os
    from pathlib import Path
    from jobflow import run_locally

    # test if same band structure gives out expected 0.0 rmse
    dummy_bs_file_path = test_dir / "benchmark" / "NaCl.json"

    with open(dummy_bs_file_path, "r") as file:
        dummy_bs_dict = json.load(file)

    parent_dir = os.getcwd()

    os.chdir(test_dir / "benchmark")
    df_bs = PhononBandStructureSymmLine.from_dict(dummy_bs_dict)
    ml_bs = PhononBandStructureSymmLine.from_dict(dummy_bs_dict)

    benchmark_job = compute_bandstructure_benchmark_metrics(
        structure=df_bs.structure, ml_phonon_bs=ml_bs, dft_phonon_bs=df_bs
    )

    responses = run_locally(benchmark_job, create_folders=False, ensure_success=True)

    assert responses[benchmark_job.output.uuid][1].output == pytest.approx(0.0)

    # get list of generated plot files
    test_files_dir = Path(test_dir / "benchmark").resolve()
    path_to_plot_files = list(test_files_dir.glob("NaCl*.eps"))

    # ensure two plots are generated
    assert len(path_to_plot_files) == 2

    # remove the plot files from directory
    for file in path_to_plot_files:
        file.unlink()

    os.chdir(parent_dir)


def test_compute_bandstructure_benchmark_metrics(test_dir, clean_dir):
    import os
    from pathlib import Path
    from jobflow import run_locally

    # test wih two different band-structures
    dft_bs_file_path = test_dir / "benchmark" / "DFT_phonon_band_structure.yaml"
    ml_bs_file_path = test_dir / "benchmark" / "GAP_phonon_band_structure.yaml"

    parent_dir = os.getcwd()

    os.chdir(test_dir / "benchmark")
    df_bs = get_ph_bs_symm_line(bands_path=dft_bs_file_path)
    ml_bs = get_ph_bs_symm_line(bands_path=ml_bs_file_path)

    benchmark_job = compute_bandstructure_benchmark_metrics(
        structure=df_bs.structure, ml_phonon_bs=ml_bs, dft_phonon_bs=df_bs
    )

    responses = run_locally(benchmark_job, create_folders=False, ensure_success=True)

    assert responses[benchmark_job.output.uuid][1].output == pytest.approx(
        0.5716963823412201, abs=0.3  # TODO check results
    )

    # get list of generated plot files
    test_files_dir = Path(test_dir / "benchmark").resolve()
    path_to_plot_files = list(test_files_dir.glob("LiCl*.eps"))

    # ensure two plots are generated
    assert len(path_to_plot_files) == 2
    # remove the plot files from directory
    for file in path_to_plot_files:
        file.unlink()

    os.chdir(parent_dir)


def test_write_benchmark_metrics(test_dir, clean_dir):
    import os
    from jobflow import run_locally

    parent_dir = os.getcwd()

    os.chdir(test_dir / "benchmark")

    path_to_struct = test_dir / "benchmark" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    write_metrics_job = write_benchmark_metrics(
        benchmark_structure=structure,
        rmse=[1.2014670270901717],
        displacements=[0.01],
        mp_id="mp-22905",
    )

    _ = run_locally(write_metrics_job, create_folders=False, ensure_success=True)

    # get list of generated txt file
    path_to_ref_txt_file = test_dir / "benchmark" / "results_LiCl_ref.txt"
    path_to_txt_file = test_dir / "benchmark" / "results_LiCl.txt"

    with open(path_to_ref_txt_file, "r") as ref:
        ref_txt = ref.readlines()

    with open(path_to_txt_file, "r") as org:
        org_txt = org.readlines()

    assert ref_txt == org_txt

    # remove generated file as part of test
    path_to_txt_file.unlink()

    os.chdir(parent_dir)
