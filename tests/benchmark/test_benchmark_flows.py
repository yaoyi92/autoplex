from __future__ import annotations

import pytest
from autoplex.benchmark.phonons.flows import PhononBenchmarkMaker
from pymatgen.io.phonopy import get_ph_bs_symm_line


def test_benchmark(test_dir, clean_dir):
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

    benchmark_flow = PhononBenchmarkMaker().make(
        structure=df_bs.structure,
        ml_phonon_bs=ml_bs,
        dft_phonon_bs=df_bs,
        benchmark_mp_id="test",
    )
    assert len(benchmark_flow.jobs) == 1

    responses = run_locally(benchmark_flow, create_folders=False, ensure_success=True)

    assert responses[benchmark_flow.output.uuid][1].output == pytest.approx(
        0.5716963823412201, abs=0.02
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
