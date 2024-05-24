from __future__ import annotations


from autoplex.fitting.common.flows import MLIPFitMaker


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
        "IsolatedAtom": {"iso_atoms_dir": [[
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
    gapfit = MLIPFitMaker().make(species_list=["Li", "Cl"], isolated_atoms_energy=[-0.28649227, -0.25638457],
                                 fit_input=fit_input_dict)

    responses = run_locally(
        gapfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    test_files_dir = Path(test_dir / "fitting").resolve()
    path_to_job_files = list(test_files_dir.glob("job*"))
    # check if gap fit file is generated
    assert Path(gapfit.output["mlip_path"].resolve(memory_jobstore)).exists()

    for job_dir in path_to_job_files:
        shutil.rmtree(job_dir)

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
        "IsolatedAtom": {"iso_atoms_dir": [[
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
    gapfit = MLIPFitMaker().make(species_list=["Li", "Cl"], isolated_atoms_energy=[-0.28649227, -0.25638457],
                                 fit_input=fit_input_dict, auto_delta=False, glue_xml=False, **{
            "twob": {"delta": 2.0, "cutoff": 8}, "threeb": {"n_sparse": 100},
            "split_ratio": 0.4, "regularization": False, "distillation": True, "f_max": 40,
        })

    responses = run_locally(
        gapfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    test_files_dir = Path(test_dir / "fitting").resolve()
    path_to_job_files = list(test_files_dir.glob("job*"))

    # check if gap fit file is generated
    assert Path(gapfit.output["mlip_path"].resolve(memory_jobstore)).exists()

    for job_dir in path_to_job_files:
        shutil.rmtree(job_dir)

    os.chdir(parent_dir)

def test_mlip_fit_maker_with_pre_database_dir(test_dir, memory_jobstore, vasp_test_dir, clean_dir):
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
        "IsolatedAtom": {"iso_atoms_dir": [[
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

    test_files_dir = Path(test_dir / "fitting").resolve()

    # # Test to check if gap fit runs with pre_database_dir
    # gapfit = MLIPFitMaker().make(species_list=["Li", "Cl"], isolated_atoms_energy=[-0.28649227, -0.25638457],
    #                              fit_input=fit_input_dict, pre_database_dir=str(test_files_dir),
    #                              pre_xyz_files=["pre_xyz_train.extxyz", "pre_xyz_test.extxyz"])

    # responses = run_locally(
    #     gapfit, ensure_success=True, create_folders=True, store=memory_jobstore
    # )

    # path_to_job_files = list(test_files_dir.glob("job*"))

    # # check if gap fit file is generated
    # assert Path(gapfit.output["mlip_path"].resolve(memory_jobstore)).exists()

    # for job_dir in path_to_job_files:
    #     shutil.rmtree(job_dir)

    # os.chdir(parent_dir)


    # # Test to check if julia-ace fit runs with pre_database_dir
    # jacefit = MLIPFitMaker(mlip_type="J-ACE", mlip_hyper={'order':3, 'totaldegree':6, 'cutoff':2.0,
    #                              'solver':'BLR',}).make(isol_es={3: -0.28649227, 17:-0.25638457},
    #                              fit_input=fit_input_dict, pre_database_dir=str(test_files_dir),
    #                              pre_xyz_files=["pre_xyz_train.extxyz", "pre_xyz_test.extxyz"],num_processes=4)

    # responses = run_locally(
    #     jacefit, ensure_success=True, create_folders=True, store=memory_jobstore
    # )

    # path_to_job_files = list(test_files_dir.glob("job*"))

    # # check if gap fit file is generated
    # assert Path(jacefit.output["mlip_path"].resolve(memory_jobstore)).exists()

    # for job_dir in path_to_job_files:
    #     shutil.rmtree(job_dir)

    # os.chdir(parent_dir)


    # # Test to check if nequip fit runs with pre_database_dir
    # nequipfit = MLIPFitMaker(mlip_type="NEQUIP", mlip_hyper = {
    #             "r_max": 4.0,
    #             "num_layers": 4,
    #             "l_max": 2,
    #             "num_features": 32,
    #             "num_basis": 8,
    #             "invariant_layers": 2,
    #             "invariant_neurons": 64,
    #             "batch_size": 1,
    #             "learning_rate": 0.005,
    #             "max_epochs": 4,
    #             "default_dtype": "float32",
    #             "device": 'cpu',
    #         }).make(fit_input=fit_input_dict,
    #                             isol_es={3: -0.28649227, 17:-0.25638457},
    #                             pre_database_dir=str(test_files_dir),
    #                             pre_xyz_files=["pre_xyz_train.extxyz", "pre_xyz_test.extxyz"],
    #                             num_processes=1)

    # responses = run_locally(
    #     nequipfit, ensure_success=True, create_folders=True, store=memory_jobstore
    # )

    # path_to_job_files = list(test_files_dir.glob("job*"))

    # # check if gap fit file is generated
    # assert Path(nequipfit.output["mlip_path"].resolve(memory_jobstore)).exists()

    # for job_dir in path_to_job_files:
    #     shutil.rmtree(job_dir)

    # os.chdir(parent_dir)


    # Test to check if m3gnet fit runs with pre_database_dir
    m3gnetfit = MLIPFitMaker(mlip_type="M3GNET", mlip_hyper = {
                "exp_name": "training",
                "results_dir": "m3gnet_results",
                "cutoff": 3.0,
                "threebody_cutoff": 2.0,
                "batch_size": 1,
                "max_epochs": 3,
                "include_stresses": True,
                "hidden_dim": 8,
                "num_units": 8,
                "max_l": 4,
                "max_n": 4,
                "device": "cpu",
                "test_equal_to_val": True,
            }).make(fit_input=fit_input_dict, 
                                isol_es={3: -0.28649227, 17:-0.25638457},
                                pre_database_dir=str(test_files_dir),
                                pre_xyz_files=["pre_xyz_train.extxyz", "pre_xyz_test.extxyz"],
                                num_processes=1)

    responses = run_locally(
        m3gnetfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    path_to_job_files = list(test_files_dir.glob("job*"))

    # check if gap fit file is generated
    assert Path(m3gnetfit.output["mlip_path"].resolve(memory_jobstore)).exists()

    for job_dir in path_to_job_files:
        shutil.rmtree(job_dir)

    os.chdir(parent_dir)


    # # Test to check if MACE fit runs with pre_database_dir
    # macefit = MLIPFitMaker(mlip_type="MACE", mlip_hyper = {
    #             "model": "MACE",
    #             "config_type_weights": '{"Default":1.0}',
    #             "hidden_irreps": '32x0e + 32x1o',
    #             "r_max": 3.0,
    #             "batch_size": 5,
    #             "max_num_epochs": 10,
    #             "start_swa": 5,
    #             "ema_decay": 0.99,
    #             "correlation": 3,
    #             "loss": "huber",
    #             "default_dtype": "float32",
    #             "device": "cpu",
    #         }).make(fit_input=fit_input_dict,
    #                             pre_database_dir=str(test_files_dir),
    #                             pre_xyz_files=["pre_xyz_train.extxyz", "pre_xyz_test.extxyz"],
    #                             num_processes=1)

    # responses = run_locally(
    #     macefit, ensure_success=True, create_folders=True, store=memory_jobstore
    # )

    # path_to_job_files = list(test_files_dir.glob("job*"))

    # # check if gap fit file is generated
    # assert Path(macefit.output["mlip_path"].resolve(memory_jobstore)).exists()

    # for job_dir in path_to_job_files:
    #     shutil.rmtree(job_dir)

    # os.chdir(parent_dir)

def test_mlip_fit_maker_glue_xml(
    test_dir, memory_jobstore, vasp_test_dir, clean_dir
):
    import os
    import shutil
    from pathlib import Path
    from jobflow import run_locally

    parent_dir = os.getcwd()

    os.chdir(test_dir / "fitting")

    fit_input_dict = {
        "mp-149": {
            "rand_struc_dir": [
                [
                    (
                            vasp_test_dir
                            / "Si_glue_xml_fit"
                            / "rattled_supercell_1"
                    )
                    .absolute()
                    .as_posix(),
                    (
                            vasp_test_dir
                            / "Si_glue_xml_fit"
                            / "rattled_supercell_2"
                    )
                    .absolute()
                    .as_posix(),
                    (
                            vasp_test_dir
                            / "Si_glue_xml_fit"
                            / "rattled_supercell_3"
                    )
                    .absolute()
                    .as_posix(),
                    (
                            vasp_test_dir
                            / "Si_glue_xml_fit"
                            / "rattled_supercell_4"
                    )
                    .absolute()
                    .as_posix(),
                    (
                            vasp_test_dir
                            / "Si_glue_xml_fit"
                            / "rattled_supercell_5"
                    )
                    .absolute()
                    .as_posix(),
                ]
            ],
            "phonon_dir": [
                [
                    (
                            vasp_test_dir
                            / "Si_glue_xml_fit"
                            / "phonon_supercell_1"
                    )
                    .absolute()
                    .as_posix(),
                    (
                            vasp_test_dir
                            / "Si_glue_xml_fit"
                            / "phonon_supercell_2"
                    )
                    .absolute()
                    .as_posix(),
                    (
                            vasp_test_dir
                            / "Si_glue_xml_fit"
                            / "phonon_supercell_3"
                    )
                    .absolute()
                    .as_posix(),
                    (
                            vasp_test_dir
                            / "Si_glue_xml_fit"
                            / "phonon_supercell_4"
                    )
                    .absolute()
                    .as_posix(),
                    (
                            vasp_test_dir
                            / "Si_glue_xml_fit"
                            / "phonon_supercell_5"
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
                    / "Si_glue_xml_fit"
                    / "iso_atom"
            )
            .absolute()
            .as_posix(),
        ]]
        }
    }

    # Test to check if gap fit runs with default hyperparameter sets (i.e. include_two_body and include_soap is True)
    gapfit = MLIPFitMaker().make(species_list=["Si"], isolated_atoms_energy=[-0.82067307], fit_input=fit_input_dict,
                                 auto_delta=False, glue_xml=True, **{
            "gap_para": {"two_body": False, "three_body": False, "soap": True},
             "general": {"core_param_file": "glue.xml", "core_ip_args": "{IP Glue}"}
        })

    responses = run_locally(
        gapfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    test_files_dir = Path(test_dir / "fitting").resolve()
    path_to_job_files = list(test_files_dir.glob("job*"))

    # check if gap fit file is generated
    assert Path(gapfit.output["mlip_path"].resolve(memory_jobstore)).exists()

    for job_dir in path_to_job_files:
        shutil.rmtree(job_dir)

    os.chdir(parent_dir)
