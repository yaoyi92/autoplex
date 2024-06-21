"""fitting using GAP."""
from __future__ import annotations

from pathlib import Path
from jobflow import job
from ase.io import read, write
from autoplex.fitting.common.utils import (
    ace_fitting,
    check_convergence,
    gap_fitting,
    m3gnet_fitting,
    mace_fitting,
    nequip_fitting,
    stratified_dataset_split,
    data_distillation,
)
from autoplex.fitting.common.regularization import set_sigma
import shutil
import os

current_dir = Path(__file__).absolute().parent
GAP_DEFAULTS_FILE_PATH = current_dir / "gap-defaults.json"


@job
def machine_learning_fit(
    database_dir: str,
    species_list: list,
    isol_es: dict | None = None,
    num_processes: int = 32,
    auto_delta: bool = True,
    glue_xml: bool = False,
    mlip_type: str | None = None,
    regularization: bool = True,
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
    mlip_hyper: dict
        basic MLIP hyperparameters
    regularization: bool
        For using sigma regularization.
    species_list : list.
            List of element names (str)
    HPO: bool
        call hyperparameter optimization (HPO) or not
    kwargs : dict.
            dict including more fit keyword args.
    """
    train_files = [
        "train.extxyz",
        "train_wo_sigma.extxyz",
        "train_phonon.extxyz",
        "train_rand_struc.extxyz",
    ]
    test_files = [
        "test.extxyz",
        "test.extxyz",
        "test_phonon.extxyz",
        "test_rand_struc.extxyz",
    ]

    if mlip_type == "GAP":
        defult_mlip_hyper = {"two_body": True, "three_body": False, "soap": True}

    elif mlip_type == "J-ACE":
        defult_mlip_hyper = {"order": 3, "totaldegree": 6, "cutoff": 2.0, "solver": "BLR"}

    elif mlip_type == "NEQUIP":
        defult_mlip_hyper = {
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
        defult_mlip_hyper = {
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
        defult_mlip_hyper = {
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

    if mlip_hyper is not None:

        defult_mlip_hyper.update(mlip_hyper)

    mlip_hyper = defult_mlip_hyper

    if mlip_type == "GAP":
        for train_name, test_name in zip(train_files, test_files):
            if (
                Path(Path(database_dir) / train_name).exists()
                and Path(Path(database_dir) / test_name).exists()
            ):
                train_test_error = gap_fitting(
                    db_dir=database_dir,
                    species_list=species_list,
                    include_two_body=mlip_hyper["two_body"],
                    include_three_body=mlip_hyper["three_body"],
                    include_soap=mlip_hyper["soap"],
                    num_processes=num_processes,
                    auto_delta=auto_delta,
                    glue_xml=glue_xml,
                    regularization=regularization,
                    train_name=train_name,
                    test_name=test_name,
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
            fit_kwargs=kwargs,
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
        "train_error": train_test_error["train_error"],
        "test_error": train_test_error["test_error"],
        "convergence": check_conv,
    }


@job
def data_preprocessing(split_ratio: float = 0.5,
                       regularization: bool = False,
                       distillation: bool = False,
                       f_max: float = 40.0,
                       vasp_ref_dir: str = None,
                       pre_database_dir: str = None):

    # reject strucutres with large force components
    if distillation:
        atoms = data_distillation(vasp_ref_dir, f_max)
    else:
        atoms = read(vasp_ref_dir, index=':')

    # split dataset into training and testing datasets with a ratio of 9:1
    train_structures, test_structures = stratified_dataset_split(atoms, split_ratio)

    # Merging database
    if pre_database_dir and os.path.exists(pre_database_dir):
        files_to_copy = ['train.extxyz', 'test.extxyz']
        current_working_directory = os.getcwd()

        for file_name in files_to_copy:
            source_file_path = os.path.join(pre_database_dir, file_name)
            destination_file_path = os.path.join(current_working_directory, file_name)
            shutil.copy(source_file_path, destination_file_path)
            print(f"File {file_name} has been copied to {destination_file_path}")

    write('train.extxyz', train_structures, format='extxyz', append=True)
    write('test.extxyz', test_structures, format='extxyz', append=True)

    if regularization:
        atoms = read('train.extxyz', index=':')
        atom_with_sigma = set_sigma(atoms, etup = [(0.1, 1), (0.001, 0.1), (0.0316, 0.316), (0.0632, 0.632)])
        write('train_with_sigma.extxyz',atom_with_sigma,format='extxyz')

    database_path = Path.cwd()

    return database_path