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
    gap_para: dict = gap_para,
    j-ace_para: dict = j-ace_para,
    nequip_para: dict = nequip_para,
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
    if gap_para is None:
        gap_para = {"two_body": True, "three_body": False, "soap": True}

    if mlip_type == "GAP":
        train_test_error = gap_fitting(
            db_dir=database_dir,
            include_two_body=gap_para["two_body"],
            include_three_body=gap_para["three_body"],
            include_soap=gap_para["soap"],
            num_processes=num_processes,
            auto_delta=auto_delta,
            glue_xml=glue_xml,
            fit_kwargs=kwargs,
        )

    if self.mlip_type == 'ACE':
        train_test_error = ace_fitting(dir=database_dir, 
                                      energy_name=ace_para['energy_name'], 
                                      force_name=ace_para['force_name'], 
                                      virial_name=ace_para['virial_name'],
                                      order=ace_para['order'],
                                      totaldegree=ace_para['totaldegree'],
                                      cutoff=ace_para['cutoff'],
                                      solver=ace_para['solver'],
                                      isol_es=isol_es,
                                      num_of_threads=num_of_threads)

    if self.mlip_type == 'NEQUIP':
        train_test_error = nequip_fitting(dir=database_dir,
                                         r_max=nequip_para['r_max'],
                                         num_layers=nequip_para['num_layers'],
                                         l_max=nequip_para['l_max'],
                                         num_features=nequip_para['num_features'],
                                         num_basis=nequip_para['num_basis'],
                                         invariant_layers=nequip_para['invariant_layers'],
                                         invariant_neurons=nequip_para['invariant_neurons'],
                                         batch_size=nequip_para['batch_size'],
                                         learning_rate=nequip_para['learning_rate'],
                                         default_dtype=nequip_para['default_dtype'])

    check_conv = check_convergence(train_test_error["test_error"])

    return {
        "mlip_path": train_test_error["mlip_path"],
        "mlip_xml": train_test_error["mlip_path"].joinpath("gap_file.xml"),
        "train_error": train_test_error["train_error"],
        "test_error": train_test_error["test_error"],
        "convergence": check_conv,
    }
