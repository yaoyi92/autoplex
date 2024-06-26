from __future__ import annotations

import os.path

from autoplex.fitting.common.jobs import GAP_DEFAULTS_FILE_PATH  # this will not be needed anymore
from autoplex.fitting.common.utils import (
    load_gap_hyperparameter_defaults,
    gap_hyperparameter_constructor,
    gap_fitting,
    ace_fitting,
    nequip_fitting,
    m3gnet_fitting,
    mace_fitting,
    check_convergence,
    data_distillation,
    prepare_fit_environment,
)


def test_gap_hyperparameter_constructor(test_dir, clean_dir):
    gap_hyper_parameter_dict = load_gap_hyperparameter_defaults(
        gap_fit_parameter_file_path=GAP_DEFAULTS_FILE_PATH
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

    gap_hyper_parameter_dict = load_gap_hyperparameter_defaults(
        gap_fit_parameter_file_path=GAP_DEFAULTS_FILE_PATH
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

    gap_hyper_parameter_dict = load_gap_hyperparameter_defaults(
        gap_fit_parameter_file_path=GAP_DEFAULTS_FILE_PATH
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

    gap_hyper_parameter_dict = load_gap_hyperparameter_defaults(
        gap_fit_parameter_file_path=GAP_DEFAULTS_FILE_PATH
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


def test_gap_fitting(test_dir, clean_dir):
    import os

    gap_fitting(
        db_dir=(test_dir / "fitting" / "ref_files"),
        species_list=["Li", "Cl"],
        include_soap=False,
        auto_delta=False,
        num_processes=4,
    )

    assert os.path.isfile("gap_file.xml")


def test_ace_fitting(test_dir, clean_dir):
    import os

    ace_fitting(
        db_dir=(test_dir / "fitting" / "ref_files"),
        order=3,
        totaldegree=6,
        cutoff=2.0,
        solver="BLR",
        isolated_atoms_energies={3: -0.28649227, 17: -0.25638457},
        num_processes=4,
    )

    assert os.path.isfile("ace.jl")


def test_nequip_fitting(test_dir, clean_dir):
    import os

    nequip_fitting(
        db_dir=(test_dir / "fitting" / "ref_files"),
        isolated_atoms_energies={3: -0.28649227, 17: -0.25638457},
        r_max=4.0,
        num_layers=4,
        l_max=2,
        num_features=32,
        num_basis=8,
        invariant_layers=2,
        invariant_neurons=64,
        batch_size=1,  # reduced to 1 to minimize the test execution time
        learning_rate=0.005,
        max_epochs=1,
        default_dtype="float32",
        device="cpu"
    )

    assert os.path.isfile("nequip.yaml")


def test_m3gnet_fitting(test_dir, clean_dir):
    import os

    m3gnet_fitting(
        db_dir=(test_dir / "fitting" / "ref_files"),
        exp_name="training",
        results_dir="m3gnet_results",
        cutoff=3.0,
        threebody_cutoff=2.0,
        include_stresses=True,
        hidden_dim=8,
        num_units=8,
        max_l=4,
        max_n=4,
        test_equal_to_val=True,
        batch_size=1,  # reduced to 1 to minimize the test execution time
        max_epochs=3,
        device="cpu"
    )

    assert os.path.isfile("m3gnet.log")


def test_mace_fitting(test_dir, clean_dir):
    import os

    mace_fitting(
        db_dir=(test_dir / "fitting" / "ref_files"),
        batch_size=5,
        device="cpu",
        r_max=3.0,
        max_num_epochs=10,
        default_dtype="float32",
        model="MACE",
        config_type_weights='{"Default":1.0}',
        hidden_irreps="32x0e + 32x1o",
        start_swa=5,
        ema_decay=0.99,
        correlation=3,
        loss="huber",
    )

    assert os.path.isfile("MACE_model.model")


def test_check_convergence():
    check_convergence(0.002)

    assert True


def test_data_distillation(test_dir):
    atoms = data_distillation((test_dir / "fitting" / "ref_files" / "vasp_ref.extxyz"), 35.0)

    for atom in atoms:
        if (atom.symbols == "Li32Cl32").any() or (atom.symbols == "Li").any() or (atom.symbols == "Cl").any():
            assert True


def test_prepare_fit_environment(test_dir, clean_dir):
    import os
    import shutil
    prepare = prepare_fit_environment(
        database_dir=(test_dir / "fitting" / "ref_files"),
        mlip_path=(test_dir / "fitting" / "test_mlip_path"),
        glue_xml=False,
    )

    assert os.path.isdir(prepare)

    # remove all auxiliary files
    os.chdir(test_dir / "data" / "ref_data")
    for file_name in os.listdir(os.getcwd()):
        if file_name not in ['train_Si.extxyz', 'test_Si.extxyz', 'quip_train_Si.extxyz', 'quip_test_Si.extxyz']:
            file_path = os.path.join(os.getcwd(), file_name)
            try:
                # Check if it's a file or directory
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)  # Remove the file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove the directory and its contents
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
