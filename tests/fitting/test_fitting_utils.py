from __future__ import annotations

import os.path

from autoplex.fitting.common.jobs import GAP_DEFAULTS_FILE_PATH  # this will not be needed anymore
from autoplex.fitting.common.utils import (
    load_mlip_hyperparameter_defaults,
    gap_hyperparameter_constructor,
    check_convergence,
    data_distillation,
    prepare_fit_environment,
)
from autoplex.fitting.common.flows import MLIPFitMaker
import shutil
from pathlib import Path
from jobflow import run_locally


def test_gap_hyperparameter_constructor(test_dir, clean_dir):
    gap_hyper_parameter_dict = load_mlip_hyperparameter_defaults(
        mlip_fit_parameter_file_path=GAP_DEFAULTS_FILE_PATH
    )

    gap_input_list = gap_hyperparameter_constructor(
        gap_parameter_dict=gap_hyper_parameter_dict,
        include_three_body=True,
        include_soap=True,
        include_two_body=True,
    )
    # test if string for all possible args true
    ref_list = [
        "at_file=train.extxyz",
        "default_sigma={0.0001 0.05 0.05 0}",
        "energy_parameter_name=REF_energy",
        "force_parameter_name=REF_forces",
        "virial_parameter_name=REF_virial",
        "sparse_jitter=1e-08",
        "do_copy_at_file=F",
        "openmp_chunk_size=10000",
        "gp_file=gap_file.xml",
        "gap={distance_Nb order=2 f0=0.0 add_species=T cutoff=5.0 "
        "n_sparse=15 covariance_type=ard_se delta=2.0 theta_uniform=0.5 "
        "sparse_method=uniform compact_clusters=T :distance_Nb order=3 f0=0.0 add_species=T "
        "cutoff=3.25 n_sparse=100 covariance_type=ard_se "
        "delta=2.0 theta_uniform=1.0 sparse_method=uniform compact_clusters=T :soap add_species=T "
        "l_max=10 n_max=12 atom_sigma=0.5 zeta=4 cutoff=5.0 "
        "cutoff_transition_width=0.5 central_weight=1.0 n_sparse=6000 delta=0.5 "
        "f0=0.0 covariance_type=dot_product sparse_method=cur_points}",
    ]

    assert ref_list == gap_input_list

    gap_hyper_parameter_dict = load_mlip_hyperparameter_defaults(
        mlip_fit_parameter_file_path=GAP_DEFAULTS_FILE_PATH
    )

    gap_input_list = gap_hyperparameter_constructor(
        gap_parameter_dict=gap_hyper_parameter_dict,
        include_three_body=True,
        include_soap=False,
        include_two_body=True,
    )
    # test if string for include_soap==False
    ref_list = [
        "at_file=train.extxyz",
        "default_sigma={0.0001 0.05 0.05 0}",
        "energy_parameter_name=REF_energy",
        "force_parameter_name=REF_forces",
        "virial_parameter_name=REF_virial",
        "sparse_jitter=1e-08",
        "do_copy_at_file=F",
        "openmp_chunk_size=10000",
        "gp_file=gap_file.xml",
        "gap={distance_Nb order=2 f0=0.0 add_species=T cutoff=5.0 "
        "n_sparse=15 covariance_type=ard_se delta=2.0 theta_uniform=0.5 "
        "sparse_method=uniform compact_clusters=T :distance_Nb order=3 f0=0.0 add_species=T "
        "cutoff=3.25 n_sparse=100 covariance_type=ard_se "
        "delta=2.0 theta_uniform=1.0 sparse_method=uniform compact_clusters=T}",
    ]

    assert ref_list == gap_input_list

    # test if returned string is changed if passed in dict is updated
    gap_hyper_parameter_dict["twob"].update({"cutoff": 8})
    gap_hyper_parameter_dict["threeb"].update({"cutoff": 8, "n_sparse": 100})
    gap_hyper_parameter_dict["soap"].update({"delta": 1.5, "zeta": 2})

    gap_input_list_updated = gap_hyperparameter_constructor(
        gap_parameter_dict=gap_hyper_parameter_dict,
        include_three_body=True,
        include_soap=True,
        include_two_body=True,
    )

    ref_list_exp = [
        "at_file=train.extxyz",
        "default_sigma={0.0001 0.05 0.05 0}",
        "energy_parameter_name=REF_energy",
        "force_parameter_name=REF_forces",
        "virial_parameter_name=REF_virial",
        "sparse_jitter=1e-08",
        "do_copy_at_file=F",
        "openmp_chunk_size=10000",
        "gp_file=gap_file.xml",
        "gap={distance_Nb order=2 f0=0.0 add_species=T cutoff=8 "
        "n_sparse=15 covariance_type=ard_se delta=2.0 theta_uniform=0.5 "
        "sparse_method=uniform compact_clusters=T :distance_Nb order=3 f0=0.0 add_species=T "
        "cutoff=8 n_sparse=100 covariance_type=ard_se "
        "delta=2.0 theta_uniform=1.0 sparse_method=uniform compact_clusters=T :soap "
        "add_species=T l_max=10 n_max=12 atom_sigma=0.5 zeta=2 cutoff=5.0 "
        "cutoff_transition_width=0.5 central_weight=1.0 n_sparse=6000 delta=1.5 "
        "f0=0.0 covariance_type=dot_product sparse_method=cur_points}",
    ]

    assert ref_list_exp == gap_input_list_updated

    # check disable three_body and two_body

    gap_hyper_parameter_dict = load_mlip_hyperparameter_defaults(
        mlip_fit_parameter_file_path=GAP_DEFAULTS_FILE_PATH
    )

    # three_body_disable

    gap_input_list = gap_hyperparameter_constructor(
        gap_parameter_dict=gap_hyper_parameter_dict,
        include_three_body=False,
        include_soap=True,
        include_two_body=True,
    )

    ref_list = [
        "at_file=train.extxyz",
        "default_sigma={0.0001 0.05 0.05 0}",
        "energy_parameter_name=REF_energy",
        "force_parameter_name=REF_forces",
        "virial_parameter_name=REF_virial",
        "sparse_jitter=1e-08",
        "do_copy_at_file=F",
        "openmp_chunk_size=10000",
        "gp_file=gap_file.xml",
        "gap={distance_Nb order=2 f0=0.0 add_species=T cutoff=5.0 "
        "n_sparse=15 covariance_type=ard_se delta=2.0 theta_uniform=0.5 "
        "sparse_method=uniform compact_clusters=T :soap "
        "add_species=T l_max=10 n_max=12 atom_sigma=0.5 zeta=4 cutoff=5.0 "
        "cutoff_transition_width=0.5 central_weight=1.0 n_sparse=6000 delta=0.5 "
        "f0=0.0 covariance_type=dot_product sparse_method=cur_points}",
    ]

    assert ref_list == gap_input_list

    # disable two_body

    gap_input_list = gap_hyperparameter_constructor(
        gap_parameter_dict=gap_hyper_parameter_dict,
        include_three_body=True,
        include_soap=True,
        include_two_body=False,
    )

    ref_list = [
        "at_file=train.extxyz",
        "default_sigma={0.0001 0.05 0.05 0}",
        "energy_parameter_name=REF_energy",
        "force_parameter_name=REF_forces",
        "virial_parameter_name=REF_virial",
        "sparse_jitter=1e-08",
        "do_copy_at_file=F",
        "openmp_chunk_size=10000",
        "gp_file=gap_file.xml",
        "gap={distance_Nb order=3 f0=0.0 add_species=T "
        "cutoff=3.25 n_sparse=100 covariance_type=ard_se "
        "delta=2.0 theta_uniform=1.0 sparse_method=uniform compact_clusters=T :soap "
        "add_species=T l_max=10 n_max=12 atom_sigma=0.5 zeta=4 cutoff=5.0 "
        "cutoff_transition_width=0.5 central_weight=1.0 n_sparse=6000 delta=0.5 "
        "f0=0.0 covariance_type=dot_product sparse_method=cur_points}",
    ]

    assert ref_list == gap_input_list

    gap_hyper_parameter_dict = load_mlip_hyperparameter_defaults(
        mlip_fit_parameter_file_path=GAP_DEFAULTS_FILE_PATH
    )

    # check with only soap enabled

    gap_input_list = gap_hyperparameter_constructor(
        gap_parameter_dict=gap_hyper_parameter_dict,
        include_three_body=False,
        include_soap=True,
        include_two_body=False,
    )

    ref_list = [
        "at_file=train.extxyz",
        "default_sigma={0.0001 0.05 0.05 0}",
        "energy_parameter_name=REF_energy",
        "force_parameter_name=REF_forces",
        "virial_parameter_name=REF_virial",
        "sparse_jitter=1e-08",
        "do_copy_at_file=F",
        "openmp_chunk_size=10000",
        "gp_file=gap_file.xml",
        "gap={soap "
        "add_species=T l_max=10 n_max=12 atom_sigma=0.5 zeta=4 cutoff=5.0 "
        "cutoff_transition_width=0.5 central_weight=1.0 n_sparse=6000 delta=0.5 "
        "f0=0.0 covariance_type=dot_product sparse_method=cur_points}",
    ]

    assert ref_list == gap_input_list


def test_check_convergence():
    check_convergence(0.002)

    assert True


def test_data_distillation(test_dir):
    atoms = data_distillation((test_dir / "fitting" / "ref_files" / "vasp_ref.extxyz"), 35.0, force_label="REF_forces")

    for atom in atoms:
        if (atom.symbols == "Li32Cl32").any() or (atom.symbols == "Li").any() or (atom.symbols == "Cl").any():
            assert True


def test_prepare_fit_environment(test_dir, clean_dir):
    prepare = prepare_fit_environment(
        database_dir=(test_dir / "fitting" / "ref_files"),
        mlip_path=(test_dir / "fitting" / "test_mlip_path"),
        glue_xml=False,
    )

    assert os.path.isdir(prepare)


def test_gap_fit_maker(test_dir, memory_jobstore):

    database_dir = test_dir / "fitting/rss_training_dataset/"

    gapfit = MLIPFitMaker().make(
        auto_delta=False,
        glue_xml=False,
        twob={"delta": 2.0, "cutoff": 4},
        threeb={"n_sparse": 10},
        preprocessing_data=False,
        database_dir=database_dir    
        )

    responses = run_locally(
        gapfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    assert Path(gapfit.output["mlip_path"].resolve(memory_jobstore)).exists()

    dir = Path('.')
    path_to_job_files = list(dir.glob("job*"))
    for path in path_to_job_files:
        shutil.rmtree(path)


def test_jace_fit_maker(test_dir, memory_jobstore):

    database_dir = test_dir / "fitting/rss_training_dataset/"

    jacefit = MLIPFitMaker(
        mlip_type="J-ACE",
    ).make(
        isolated_atoms_energies={14: -0.84696938},
        num_processes_fit=4,
        preprocessing_data=False,
        database_dir=database_dir,
        order=2,
        totaldegree=4,
    )

    responses = run_locally(
        jacefit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    assert Path(jacefit.output["mlip_path"].resolve(memory_jobstore)).exists()

    dir = Path('.')
    path_to_job_files = list(dir.glob("job*"))
    for path in path_to_job_files:
        shutil.rmtree(path)


def test_nqeuip_fit_maker(test_dir, memory_jobstore):
    database_dir = test_dir / "fitting/rss_training_dataset/"

    nequipfit = MLIPFitMaker(
       mlip_type="NEQUIP",
    ).make(
        isolated_atoms_energies={14: -0.84696938},
        num_processes_fit=1,
        preprocessing_data=False,
        database_dir=database_dir,
        r_max=3.14,
        max_epochs=10,
        device="cpu",
    )

    responses = run_locally(
        nequipfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    assert Path(nequipfit.output["mlip_path"].resolve(memory_jobstore)).exists()

    dir = Path('.')
    path_to_job_files = list(dir.glob("job*"))
    for path in path_to_job_files:
        shutil.rmtree(path)


def test_m3gnet_fit_maker(test_dir, memory_jobstore):
    database_dir = test_dir / "fitting/rss_training_dataset/"

    nequipfit = MLIPFitMaker(
            mlip_type="M3GNET",
    ).make(
        isolated_atoms_energies={14: -0.84696938},
        num_processes_fit=1,
        preprocessing_data=False,
        database_dir=database_dir,
        cutoff=3.0,
        threebody_cutoff=2.0,
        batch_size=1,
        max_epochs=3,
        include_stresses=True,
        hidden_dim=8,
        num_units=8,
        max_l=4,
        max_n=4,
        device="cpu",
        test_equal_to_val=True,
    )

    responses = run_locally(
        nequipfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    assert Path(nequipfit.output["mlip_path"].resolve(memory_jobstore)).exists()

    dir = Path('.')
    path_to_job_files = list(dir.glob("job*"))
    for path in path_to_job_files:
        shutil.rmtree(path)


def test_mace_fit_maker(test_dir, memory_jobstore):
    database_dir = test_dir / "fitting/rss_training_dataset/"

    nequipfit = MLIPFitMaker(
                mlip_type="MACE",
    ).make(
        isolated_atoms_energies={14: -0.84696938},
        num_processes_fit=1,
        preprocessing_data=False,
        database_dir=database_dir,
        model="MACE",
        config_type_weights='{"Default":1.0}',
        hidden_irreps="32x0e + 32x1o",
        r_max=3.0,
        batch_size=5,
        max_num_epochs=10,
        start_swa=5,
        ema_decay=0.99,
        correlation=3,
        loss="huber",
        default_dtype="float32",
        device="cpu",
    )

    responses = run_locally(
        nequipfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    assert Path(nequipfit.output["mlip_path"].resolve(memory_jobstore)).exists()

    dir = Path('.')
    path_to_job_files = list(dir.glob("job*"))
    for path in path_to_job_files:
        shutil.rmtree(path)
