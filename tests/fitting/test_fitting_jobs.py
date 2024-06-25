from __future__ import annotations
from autoplex.fitting.common.jobs import machine_learning_fit


def test_machine_learning_fit_gap(test_dir, memory_jobstore, clean_dir):
    from jobflow import run_locally
    from pathlib import Path

    gapfit = machine_learning_fit(
        mlip_type="GAP",
        database_dir=(test_dir / "fitting" / "ref_files"),
        species_list=["Li", "Cl"],
        isolated_atoms_energies={3: -0.28649227, 17: -0.25638457},
        num_processes=4,
    )

    responses = run_locally(
        gapfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    # check if fit file is generated
    assert Path(gapfit.output["mlip_path"].resolve(memory_jobstore)).exists()


def test_machine_learning_fit_jace(test_dir, memory_jobstore, clean_dir):
    from jobflow import run_locally
    from pathlib import Path

    jacefit = machine_learning_fit(
        mlip_type="J-ACE",
        database_dir=(test_dir / "fitting" / "ref_files"),
        species_list=["Li", "Cl"],
        isolated_atoms_energies={3: -0.28649227, 17: -0.25638457},
        num_processes=4,
    )

    responses = run_locally(
        jacefit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    # check if fit file is generated
    assert Path(jacefit.output["mlip_path"].resolve(memory_jobstore)).exists()


def test_machine_learning_fit_nequip(test_dir, memory_jobstore, clean_dir):
    from jobflow import run_locally
    from pathlib import Path

    nequipfit = machine_learning_fit(
        mlip_type="NEQUIP",
        mlip_hyper={
            "r_max": 4.0,
            "num_layers": 4,
            "l_max": 2,
            "num_features": 32,
            "num_basis": 8,
            "invariant_layers": 2,
            "invariant_neurons": 64,
            "batch_size": 1,
            "learning_rate": 0.005,
            "max_epochs": 1,  # reduced to 1 to minimize the test execution time
            "default_dtype": "float32",
            "device": "cpu",
        },
        database_dir=(test_dir / "fitting" / "ref_files"),
        species_list=["Li", "Cl"],
        isolated_atoms_energies={3: -0.28649227, 17: -0.25638457},
        num_processes=1,
    )

    responses = run_locally(
        nequipfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    # check if fit file is generated
    assert Path(nequipfit.output["mlip_path"].resolve(memory_jobstore)).exists()


def test_machine_learning_fit_m3gnet(test_dir, memory_jobstore, clean_dir):
    from jobflow import run_locally
    from pathlib import Path

    m3gnetfit = machine_learning_fit(
        mlip_type="M3GNET",
        mlip_hyper={
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
        },
        database_dir=(test_dir / "fitting" / "ref_files"),
        species_list=["Li", "Cl"],
        isolated_atoms_energies={3: -0.28649227, 17: -0.25638457},
        num_processes=1,
    )

    responses = run_locally(
        m3gnetfit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    # check if fit file is generated
    assert Path(m3gnetfit.output["mlip_path"].resolve(memory_jobstore)).exists()


def test_machine_learning_fit_mace(test_dir, memory_jobstore, clean_dir):
    from jobflow import run_locally
    from pathlib import Path

    macefit = machine_learning_fit(
        mlip_type="MACE",
        mlip_hyper={
            "model": "MACE",
            "config_type_weights": '{"Default":1.0}',
            "hidden_irreps": "32x0e + 32x1o",
            "r_max": 3.0,
            "batch_size": 5,
            "max_num_epochs": 10,
            "start_swa": 5,
            "ema_decay": 0.99,
            "correlation": 3,
            "loss": "huber",
            "default_dtype": "float32",
            "device": "cpu",
        },
        database_dir=(test_dir / "fitting" / "ref_files"),
        species_list=["Li", "Cl"],
        isolated_atoms_energies={3: -0.28649227, 17: -0.25638457},
        num_processes=1,
    )

    responses = run_locally(
        macefit, ensure_success=True, create_folders=True, store=memory_jobstore
    )

    # check if fit file is generated
    assert Path(macefit.output["mlip_path"].resolve(memory_jobstore)).exists()
