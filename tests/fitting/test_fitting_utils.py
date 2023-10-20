from __future__ import annotations
from autoplex.fitting.jobs import GAP_DEFAULTS_FILE_PATH
from autoplex.fitting.utils import (
    load_gap_hyperparameter_defaults,
    gap_hyperparameter_constructor,
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
        "at_file=trainGAP.xyz",
        "e0=2",
        "default_sigma={0.01 0.2 0.2 0.0}",
        "energy_parameter_name=energy",
        "force_parameter_name=forces",
        "virial_parameter_name=virial",
        "sparse_jitter=1e-08",
        "do_copy_at_file=F",
        "openmp_chunk_size=10000",
        "gp_file=gap.xml",
        "gap={distance_Nb order=2 f0=0.0 cutoff=6.5 cutoff_transition_width=0.5 "
        "n_sparse=15 covariance_type=ard_se delta=2.0 theta_uniform=2.0 "
        "sparse_method=uniform compact_clusters=T :distance_Nb order=3 f0=0.0 "
        "cutoff=6.5 cutoff_transition_width=0.5 n_sparse=150 covariance_type=ard_se "
        "delta=2.0 theta_uniform=2.0 sparse_method=uniform compact_clusters=T :soap "
        "l_max=6 n_max=12 atom_sigma=0.5 zeta=4 cutoff=6.5 "
        "cutoff_transition_width=0.5 central_weight=1.0 n_sparse=7000 delta=0.5 "
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
        "at_file=trainGAP.xyz",
        "e0=2",
        "default_sigma={0.01 0.2 0.2 0.0}",
        "energy_parameter_name=energy",
        "force_parameter_name=forces",
        "virial_parameter_name=virial",
        "sparse_jitter=1e-08",
        "do_copy_at_file=F",
        "openmp_chunk_size=10000",
        "gp_file=gap.xml",
        "gap={distance_Nb order=2 f0=0.0 cutoff=6.5 cutoff_transition_width=0.5 "
        "n_sparse=15 covariance_type=ard_se delta=2.0 theta_uniform=2.0 "
        "sparse_method=uniform compact_clusters=T :distance_Nb order=3 f0=0.0 "
        "cutoff=6.5 cutoff_transition_width=0.5 n_sparse=150 covariance_type=ard_se "
        "delta=2.0 theta_uniform=2.0 sparse_method=uniform compact_clusters=T}",
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
        "at_file=trainGAP.xyz",
        "e0=2",
        "default_sigma={0.01 0.2 0.2 0.0}",
        "energy_parameter_name=energy",
        "force_parameter_name=forces",
        "virial_parameter_name=virial",
        "sparse_jitter=1e-08",
        "do_copy_at_file=F",
        "openmp_chunk_size=10000",
        "gp_file=gap.xml",
        "gap={distance_Nb order=2 f0=0.0 cutoff=8 cutoff_transition_width=0.5 "
        "n_sparse=15 covariance_type=ard_se delta=2.0 theta_uniform=2.0 "
        "sparse_method=uniform compact_clusters=T :distance_Nb order=3 f0=0.0 "
        "cutoff=8 cutoff_transition_width=0.5 n_sparse=100 covariance_type=ard_se "
        "delta=2.0 theta_uniform=2.0 sparse_method=uniform compact_clusters=T :soap "
        "l_max=6 n_max=12 atom_sigma=0.5 zeta=2 cutoff=6.5 "
        "cutoff_transition_width=0.5 central_weight=1.0 n_sparse=7000 delta=1.5 "
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
        "at_file=trainGAP.xyz",
        "e0=2",
        "default_sigma={0.01 0.2 0.2 0.0}",
        "energy_parameter_name=energy",
        "force_parameter_name=forces",
        "virial_parameter_name=virial",
        "sparse_jitter=1e-08",
        "do_copy_at_file=F",
        "openmp_chunk_size=10000",
        "gp_file=gap.xml",
        "gap={distance_Nb order=2 f0=0.0 cutoff=6.5 cutoff_transition_width=0.5 "
        "n_sparse=15 covariance_type=ard_se delta=2.0 theta_uniform=2.0 "
        "sparse_method=uniform compact_clusters=T :soap "
        "l_max=6 n_max=12 atom_sigma=0.5 zeta=4 cutoff=6.5 "
        "cutoff_transition_width=0.5 central_weight=1.0 n_sparse=7000 delta=0.5 "
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
        "at_file=trainGAP.xyz",
        "e0=2",
        "default_sigma={0.01 0.2 0.2 0.0}",
        "energy_parameter_name=energy",
        "force_parameter_name=forces",
        "virial_parameter_name=virial",
        "sparse_jitter=1e-08",
        "do_copy_at_file=F",
        "openmp_chunk_size=10000",
        "gp_file=gap.xml",
        "gap={distance_Nb order=3 f0=0.0 "
        "cutoff=6.5 cutoff_transition_width=0.5 n_sparse=150 covariance_type=ard_se "
        "delta=2.0 theta_uniform=2.0 sparse_method=uniform compact_clusters=T :soap "
        "l_max=6 n_max=12 atom_sigma=0.5 zeta=4 cutoff=6.5 "
        "cutoff_transition_width=0.5 central_weight=1.0 n_sparse=7000 delta=0.5 "
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
        "at_file=trainGAP.xyz",
        "e0=2",
        "default_sigma={0.01 0.2 0.2 0.0}",
        "energy_parameter_name=energy",
        "force_parameter_name=forces",
        "virial_parameter_name=virial",
        "sparse_jitter=1e-08",
        "do_copy_at_file=F",
        "openmp_chunk_size=10000",
        "gp_file=gap.xml",
        "gap={soap "
        "l_max=6 n_max=12 atom_sigma=0.5 zeta=4 cutoff=6.5 "
        "cutoff_transition_width=0.5 central_weight=1.0 n_sparse=7000 delta=0.5 "
        "f0=0.0 covariance_type=dot_product sparse_method=cur_points}",
    ]

    assert ref_list == gap_input_list
