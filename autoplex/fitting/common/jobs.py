"""fitting using GAP."""
from __future__ import annotations

from pathlib import Path

from jobflow import job

from autoplex.fitting.common.utils import check_convergence, gap_fitting, ace_fitting, nequip_fitting

current_dir = Path(__file__).absolute().parent
GAP_DEFAULTS_FILE_PATH = current_dir / "gap-defaults.json"


@job
def machine_learning_fit(
    database_dir: str,
    isol_es: None = None,
    num_processes: int = 32,
    auto_delta: bool = True,
    glue_xml: bool = False,
    mlip_type: str | None = None,
    HPO: bool = False,
    mlip_hyper: dict|None=None,
    **kwargs,
):
    """
    Maker for fitting potential(s).

    Parameters
    ----------
    database_dir: str
        the database directory.
    gap_para: dict
        gap fit parameters.
    isol_es:
        isolated es.
    num_processes: int
        number of processes for fitting.
    auto_delta: bool
        automatically determine delta for 2b, 3b and SOAP terms.
    glue_xml: bool
        use the glue.xml core potential instead of fitting 2b terms.
    kwargs: dict.
        optional dictionary with parameters for gap fitting.
    mlip_type: str
        Choose one specific MLIP type:
        'GAP' | 'ACE' | 'Nequip' | 'M3GNet' | 'MACE'
    HPO: bool
        call hyperparameter optimization (HPO) or not
    """
    if mlip_hyper is None:
        if mlip_type == "GAP":
            mlip_hyper = {"two_body": True, "three_body": False, "soap": True}
        elif mlip_type == "J-ACE":
            mlip_hyper = {'order': 3, 'totaldegree': 6, 'cutoff': 2.0, 'solver': 'BLR'}
        elif mlip_type == "NEQUIP":
            mlip_hyper = {'r_max': 4.0, 'num_layers': 4, 'l_max': 2, 'num_features': 32, 'num_basis': 8,
                          'invariant_layers': 2, 'invariant_neurons': 64, 'batch_size': 5, 'learning_rate': 0.005,
                          'default_dtype': "float32"}
    print(mlip_hyper["two_body"])
    if mlip_type == "GAP":
        train_test_error = gap_fitting(
            db_dir=database_dir,
            include_two_body=mlip_hyper["two_body"],
            include_three_body=mlip_hyper["three_body"],
            include_soap=mlip_hyper["soap"],
            num_processes=num_processes,
            auto_delta=auto_delta,
            glue_xml=glue_xml,
            fit_kwargs=kwargs,
        )

    elif mlip_type == 'ACE':
        train_test_error = ace_fitting(dir=database_dir, 
                                      energy_name=mlip_hyper['energy_name'],
                                      force_name=mlip_hyper['force_name'],
                                      virial_name=mlip_hyper['virial_name'],
                                      order=mlip_hyper['order'],
                                      totaldegree=mlip_hyper['totaldegree'],
                                      cutoff=mlip_hyper['cutoff'],
                                      solver=mlip_hyper['solver'],
                                      isol_es=isol_es,
                                      num_of_threads=num_processes)

    elif mlip_type == 'NEQUIP':
        train_test_error = nequip_fitting(dir=database_dir,
                                         r_max=mlip_hyper['r_max'],
                                         num_layers=mlip_hyper['num_layers'],
                                         l_max=mlip_hyper['l_max'],
                                         num_features=mlip_hyper['num_features'],
                                         num_basis=mlip_hyper['num_basis'],
                                         invariant_layers=mlip_hyper['invariant_layers'],
                                         invariant_neurons=mlip_hyper['invariant_neurons'],
                                         batch_size=mlip_hyper['batch_size'],
                                         learning_rate=mlip_hyper['learning_rate'],
                                         default_dtype=mlip_hyper['default_dtype'])

    check_conv = check_convergence(train_test_error["test_error"])

    return {
        "mlip_path": train_test_error["mlip_path"],
        "mlip_xml": train_test_error["mlip_path"].joinpath("gap_file.xml"),
        "train_error": train_test_error["train_error"],
        "test_error": train_test_error["test_error"],
        "convergence": check_conv,
    }
