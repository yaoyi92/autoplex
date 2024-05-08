"""fitting using GAP."""
from __future__ import annotations

from pathlib import Path

from jobflow import job

from autoplex.fitting.common.utils import (
    ace_fitting,
    check_convergence,
    gap_fitting,
    m3gnet_fitting,
    mace_fitting,
    nequip_fitting,
)

current_dir = Path(__file__).absolute().parent
GAP_DEFAULTS_FILE_PATH = current_dir / "gap-defaults.json"


@job
def machine_learning_fit(
    database_dir: str,
    isol_es: dict | None = None,
    num_processes: int = 32,
    auto_delta: bool = True,
    glue_xml: bool = False,
    mlip_type: str | None = None,
    HPO: bool = False,
    mlip_hyper: dict | None = None,
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
            mlip_hyper = {"order": 3, "totaldegree": 6, "cutoff": 2.0, "solver": "BLR"}

        elif mlip_type == "NEQUIP":
            mlip_hyper = {
                "r_max": 4.0,
                "num_layers": 4,
                "l_max": 2,
                "num_features": 32,
                "num_basis": 8,
                "invariant_layers": 2,
                "invariant_neurons": 64,
                "batch_size": 5,
                "learning_rate": 0.005,
                "max_epochs": 10000,
                "default_dtype": "float32",
                "device": "cuda",
            }

        elif mlip_type == "M3GNET":
            mlip_hyper = {
                "exp_name": "training",
                "results_dir": "m3gnet_results",
                "cutoff": 5.0,
                "threebody_cutoff": 4.0,
                "batch_size": 10,
                "max_epochs": 1000,
                "include_stresses": True,
                "hidden_dim": 128,
                "num_units": 128,
                "max_l": 4,
                "max_n": 4,
                "device": "cuda",
                "test_equal_to_val": True,
            }

        else:
            mlip_hyper = {
                "model": "MACE",
                "config_type_weights": '{"Default":1.0}',
                "hidden_irreps": "128x0e + 128x1o",
                "r_max": 5.0,
                "batch_size": 10,
                "max_num_epochs": 1500,
                "start_swa": 1200,
                "ema_decay": 0.99,
                "correlation": 3,
                "loss": "huber",
                "default_dtype": "float32",
                "device": "cuda",
            }

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

    elif mlip_type == "J-ACE":
        train_test_error = ace_fitting(
            db_dir=database_dir,
            order=mlip_hyper["order"],
            totaldegree=mlip_hyper["totaldegree"],
            cutoff=mlip_hyper["cutoff"],
            solver=mlip_hyper["solver"],
            isol_es=isol_es,
            num_processes=num_processes,
        )

    elif mlip_type == "NEQUIP":
        train_test_error = nequip_fitting(
            db_dir=database_dir,
            r_max=mlip_hyper["r_max"],
            num_layers=mlip_hyper["num_layers"],
            l_max=mlip_hyper["l_max"],
            num_features=mlip_hyper["num_features"],
            num_basis=mlip_hyper["num_basis"],
            invariant_layers=mlip_hyper["invariant_layers"],
            invariant_neurons=mlip_hyper["invariant_neurons"],
            batch_size=mlip_hyper["batch_size"],
            learning_rate=mlip_hyper["learning_rate"],
            max_epochs=mlip_hyper["max_epochs"],
            isol_es=isol_es,
            default_dtype=mlip_hyper["default_dtype"],
            device=mlip_hyper["device"],
        )

    elif mlip_type == "M3GNET":
        train_test_error = m3gnet_fitting(
            db_dir=database_dir,
            exp_name=mlip_hyper["exp_name"],
            results_dir=mlip_hyper["results_dir"],
            cutoff=mlip_hyper["cutoff"],
            threebody_cutoff=mlip_hyper["threebody_cutoff"],
            batch_size=mlip_hyper["batch_size"],
            max_epochs=mlip_hyper["max_epochs"],
            include_stresses=mlip_hyper["include_stresses"],
            hidden_dim=mlip_hyper["hidden_dim"],
            num_units=mlip_hyper["num_units"],
            max_l=mlip_hyper["max_l"],
            max_n=mlip_hyper["max_n"],
            device=mlip_hyper["device"],
            test_equal_to_val=mlip_hyper["test_equal_to_val"],
        )

    elif mlip_type == "MACE":
        train_test_error = mace_fitting(
            db_dir=database_dir,
            model=mlip_hyper["model"],
            config_type_weights=mlip_hyper["config_type_weights"],
            hidden_irreps=mlip_hyper["hidden_irreps"],
            r_max=mlip_hyper["r_max"],
            batch_size=mlip_hyper["batch_size"],
            max_num_epochs=mlip_hyper["max_num_epochs"],
            start_swa=mlip_hyper["start_swa"],
            ema_decay=mlip_hyper["ema_decay"],
            correlation=mlip_hyper["correlation"],
            loss=mlip_hyper["loss"],
            default_dtype=mlip_hyper["default_dtype"],
            device=mlip_hyper["device"],
        )

    check_conv = check_convergence(train_test_error["test_error"])

    return {
        "mlip_path": train_test_error["mlip_path"],
        "mlip_xml": train_test_error["mlip_path"].joinpath("gap_file.xml"),
        "train_error": train_test_error["train_error"],
        "test_error": train_test_error["test_error"],
        "convergence": check_conv,
    }
