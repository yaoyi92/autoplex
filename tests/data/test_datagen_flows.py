from __future__ import annotations

import os
from pymatgen.core.structure import Structure
from atomate2.vasp.powerups import update_user_incar_settings
from autoplex.data.phonons.flows import RandomStructuresDataGenerator, IsoAtomMaker
from autoplex.data.common.flows import GenerateTrainingDataForTesting

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1


def test_data_generation(vasp_test_dir, mock_vasp, clean_dir):
    from jobflow import run_locally

    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    test_mpid = "mp-22905"
    ref_paths = {
        "phonon static 1/3": "dft_ml_data_generation/rand_static_1/",
        "phonon static 2/3": "dft_ml_data_generation/rand_static_2/",
        "phonon static 3/3": "dft_ml_data_generation/rand_static_3/",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    # disabled poscar checks here to avoid failures due to randomness issues
    fake_run_vasp_kwargs = {
        "phonon static 1/3": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "kpoints", "potcar"],
        },
        "phonon static 2/3": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "kpoints", "potcar"],
        },
        "phonon static 3/3": {
            "incar_settings": ["NSW", "ISMEAR"],
            "check_inputs": ["incar", "kpoints", "potcar"],
        },
    }
    data_gen = RandomStructuresDataGenerator(n_structures=3).make(
        structure=structure, mp_id=test_mpid, volume_custom_scale_factors=[1.0]
    )

    data_gen = update_user_incar_settings(data_gen, {"ISMEAR": 0})

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(data_gen, create_folders=True, ensure_success=True)

    assert len(responses[data_gen.output[0].uuid][2].output["dirs"]) == 3
    job_names = ["phonon static 1/3", "phonon static 2/3", "phonon static 3/3"]
    for inx, name in enumerate(job_names):
        assert responses[data_gen.output[0].uuid][1].replace.jobs[inx].name == name


def test_iso_atom_maker(mock_vasp, clean_dir):
    from jobflow import run_locally
    from pymatgen.core import Species

    specie = Species("Cl")

    ref_paths = {
        "Cl-statisoatom": "Cl_iso_atoms/Cl-statisoatom/",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "Cl-statisoatom": {"incar_settings": ["NSW", "ISMEAR"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # generate the job
    job_iso = IsoAtomMaker().make(all_species=[specie])

    job_iso = update_user_incar_settings(job_iso, {"ISMEAR": 0})

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(job_iso, create_folders=True, ensure_success=True)

    assert (
            responses[job_iso.job_uuids[0]][1].output.output.energy_per_atom == -0.2563903
    )


def test_generate_training_data_for_testing(
        vasp_test_dir, test_dir, memory_jobstore, clean_dir
    ):
    from jobflow import run_locally

    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    potential_file_dir = test_dir / "fitting" / "ref_files" / "gap.xml"
    structure = Structure.from_file(path_to_struct)
    generate_data = GenerateTrainingDataForTesting().make(train_structure_list=[structure], cell_factor_sequence=[0.95, 1.0, 1.05],
                                          potential_filename=potential_file_dir, n_struct=1, steps=1)

    responses = run_locally(generate_data, create_folders=False, ensure_success=True, store=memory_jobstore)
    #TODO unit test only runs with create_Folders=False because ForceFieldTaskDocument.from_ase_compatible_result() has no attribute dir_name implemented _summary_