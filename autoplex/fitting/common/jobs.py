"""fitting using GAP."""
from __future__ import annotations

from pathlib import Path

from jobflow import job

from autoplex.fitting.common.utils import check_convergence, gap_fitting

current_dir = Path(__file__).absolute().parent
GAP_DEFAULTS_FILE_PATH = current_dir / "gap-defaults.json"


@job
def machine_learning_fit(
    database_dir: str,
    gap_para=None,
    isol_es: None = None,
    num_processes: int = 32,
    auto_delta: bool = True,
    glue_xml: bool = False,
    mlip_type: str | None = None,
    regularization: bool = True,
    HPO: bool = False,
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
        'GAP' | 'SNAP' | 'ACE' | 'Nequip' | 'Allegro' | 'MACE'
    regularization: bool
        For using sigma regularization.
    HPO: bool
        call hyperparameter optimization (HPO) or not
    """
    if gap_para is None:
        gap_para = {"two_body": True, "three_body": False, "soap": True}

    if mlip_type is None:
        raise ValueError(
            "MLIP type is not defined! "
            "The current version supports the fitting of GAP, SNAP, ACE, Nequip, Allegro, or MACE."
        )

    if mlip_type == "GAP":
        train_test_error = gap_fitting(
            db_dir=database_dir,
            include_two_body=gap_para["two_body"],
            include_three_body=gap_para["three_body"],
            include_soap=gap_para["soap"],
            num_processes=num_processes,
            auto_delta=auto_delta,
            glue_xml=glue_xml,
            regularization=regularization,
            fit_kwargs=kwargs,
        )

    check_conv = check_convergence(train_test_error["test_error"])

    return {
        "mlip_path": train_test_error["mlip_path"],
        "mlip_xml": train_test_error["mlip_path"].joinpath("gap_file.xml"),
        "train_error": train_test_error["train_error"],
        "test_error": train_test_error["test_error"],
        "convergence": check_conv,
    }
