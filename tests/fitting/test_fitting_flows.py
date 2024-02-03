from __future__ import annotations


from autoplex.fitting.flows import MLIPFitMaker


def test_mlip_fit_maker(test_dir, clean_dir, memory_jobstore, vasp_test_dir):
    import os
    import shutil
    from pathlib import Path
    from jobflow import run_locally

    parent_dir = os.getcwd()

    os.chdir(test_dir / "fitting")

    fit_input_dict = {
        "mp-22905": {
            "rand_struc_dir": [
                [
                    (
                        vasp_test_dir
                        / "dft_ml_data_generation"
                        / "rand_static_1"
                        / "outputs"
                    )
                    .absolute()
                    .as_posix(),
                    (
                        vasp_test_dir
                        / "dft_ml_data_generation"
                        / "rand_static_2"
                        / "outputs"
                    )
                    .absolute()
                    .as_posix(),
                    (
                        vasp_test_dir
                        / "dft_ml_data_generation"
                        / "rand_static_3"
                        / "outputs"
                    )
                    .absolute()
                    .as_posix(),
                ]
            ],
            "phonon_dir": [
                [
                    (
                        vasp_test_dir
                        / "dft_ml_data_generation"
                        / "phonon_static_1"
                        / "outputs"
                    )
                    .absolute()
                    .as_posix(),
                    (
                        vasp_test_dir
                        / "dft_ml_data_generation"
                        / "phonon_static_2"
                        / "outputs"
                    )
                    .absolute()
                    .as_posix(),
                ]
            ],
            "phonon_data": [],
        },
        "isolated_atom": {"iso_atoms_dir": [[
            (
                    vasp_test_dir
                    / "Li_iso_atoms"
                    / "Li-statisoatom"
                    / "outputs"
            )
            .absolute()
            .as_posix(),
            (
                    vasp_test_dir
                    / "Cl_iso_atoms"
                    / "Cl-statisoatom"
                    / "outputs"
            )
            .absolute()
            .as_posix(),
        ]]
        }
    }

    # Test to check if gap fit runs with default hyperparameter sets (i.e. include_two_body and include_soap is True)
    gapfit = MLIPFitMaker().make(
        species_list=["Li", "Cl"],
        iso_atom_energy=[-0.28649227, -0.25638457],
        fit_input=fit_input_dict
    )

    responses = run_locally(
        gapfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    test_files_dir = Path(test_dir / "fitting").resolve()
    path_to_job_files = list(test_files_dir.glob("job*"))[0]

    # check if gap fit file is generated
    assert Path(responses[gapfit.output.uuid][1].output).exists()

    shutil.rmtree(path_to_job_files)

    os.chdir(parent_dir)


def test_mlip_fit_maker_with_kwargs(
    test_dir, clean_dir, memory_jobstore, vasp_test_dir
):
    import os
    import shutil
    from pathlib import Path
    from jobflow import run_locally

    parent_dir = os.getcwd()

    os.chdir(test_dir / "fitting")

    fit_input_dict = {
        "mp-22905": {
            "rand_struc_dir": [
                [
                    (
                            vasp_test_dir
                            / "dft_ml_data_generation"
                            / "rand_static_1"
                            / "outputs"
                    )
                    .absolute()
                    .as_posix(),
                    (
                            vasp_test_dir
                            / "dft_ml_data_generation"
                            / "rand_static_2"
                            / "outputs"
                    )
                    .absolute()
                    .as_posix(),
                    (
                            vasp_test_dir
                            / "dft_ml_data_generation"
                            / "rand_static_3"
                            / "outputs"
                    )
                    .absolute()
                    .as_posix(),
                ]
            ],
            "phonon_dir": [
                [
                    (
                            vasp_test_dir
                            / "dft_ml_data_generation"
                            / "phonon_static_1"
                            / "outputs"
                    )
                    .absolute()
                    .as_posix(),
                    (
                            vasp_test_dir
                            / "dft_ml_data_generation"
                            / "phonon_static_2"
                            / "outputs"
                    )
                    .absolute()
                    .as_posix(),
                ]
            ],
            "phonon_data": [],
        },
        "isolated_atom": {"iso_atoms_dir": [[
            (
                    vasp_test_dir
                    / "Li_iso_atoms"
                    / "Li-statisoatom"
                    / "outputs"
            )
            .absolute()
            .as_posix(),
            (
                    vasp_test_dir
                    / "Cl_iso_atoms"
                    / "Cl-statisoatom"
                    / "outputs"
            )
            .absolute()
            .as_posix(),
        ]]
        }
    }

    # Test to check if gap fit runs with default hyperparameter sets (i.e. include_two_body and include_soap is True)
    gapfit = MLIPFitMaker().make(
        species_list=["Li", "Cl"],
        iso_atom_energy=[-0.28649227, -0.25638457],
        fit_input=fit_input_dict,
        **{
            "include_soap": False,
            "include_three_body": True,
            "include_two_body": True,
            "twob": {"cutoff": 8},
            "threeb": {"n_sparse": 100},
        },
    )

    responses = run_locally(
        gapfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    test_files_dir = Path(test_dir / "fitting").resolve()
    path_to_job_files = list(test_files_dir.glob("job*"))[0]

    # check if gap fit file is generated
    assert Path(responses[gapfit.output.uuid][1].output).exists()

    shutil.rmtree(path_to_job_files)

    os.chdir(parent_dir)
