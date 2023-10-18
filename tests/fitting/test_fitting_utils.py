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

    gap_input_str = gap_hyperparameter_constructor(
        gap_parameter_dict=gap_hyper_parameter_dict,
        three_body=True,
        soap=True,
        two_body=True,
    )
    # test if string is correctly fromatted
    ref_str = (
        "gap={distance_Nb order=2 f0=0.0 cutoff=6.5 cutoff_transition_width=0.5 n_sparse=15 "
        "covariance_type=ard_se delta=2.0 theta_uniform=2.0 sparse_method=uniform "
        "compact_clusters=T distance_Nb order=3 f0=0.0 cutoff=6.5 cutoff_transition_width=0.5 "
        "n_sparse=150 covariance_type=ard_se delta=2.0 theta_uniform=2.0 sparse_method=uniform "
        "compact_clusters=T :soap l_max=6 n_max=12 atom_sigma=0.5 zeta=4 cutoff=6.5 "
        "cutoff_transition_width=0.5 central_weight=1.0 n_sparse=7000 delta=0.5 f0=0.0 "
        "covariance_type=dot_product sparse_method=cur_points}"
    )

    assert ref_str == gap_input_str

    # test if returned string is changed if passed in dict is updated
    gap_hyper_parameter_dict["twob"].update({"cutoff": 8})
    gap_hyper_parameter_dict["threeb"].update({"cutoff": 8, "n_sparse": 100})
    gap_hyper_parameter_dict["soap"].update({"delta": 1.5, "zeta": 2})

    gap_input_str_updated = gap_hyperparameter_constructor(
        gap_parameter_dict=gap_hyper_parameter_dict,
        three_body=True,
        soap=True,
        two_body=True,
    )

    # assert not ref_str == gap_input_str_updated

    ref_str_exp = (
        "gap={distance_Nb order=2 f0=0.0 cutoff=8 cutoff_transition_width=0.5 n_sparse=15 "
        "covariance_type=ard_se delta=2.0 theta_uniform=2.0 sparse_method=uniform "
        "compact_clusters=T distance_Nb order=3 f0=0.0 cutoff=8 cutoff_transition_width=0.5 "
        "n_sparse=100 covariance_type=ard_se delta=2.0 theta_uniform=2.0 sparse_method=uniform "
        "compact_clusters=T :soap l_max=6 n_max=12 atom_sigma=0.5 zeta=2 cutoff=6.5 "
        "cutoff_transition_width=0.5 central_weight=1.0 n_sparse=7000 delta=1.5 f0=0.0 "
        "covariance_type=dot_product sparse_method=cur_points}"
    )

    assert ref_str_exp == gap_input_str_updated
    print(ref_str_exp)
