from __future__ import annotations

from pymatgen.core.structure import Structure
from atomate2.vasp.powerups import update_user_incar_settings
from autoplex.data.flows import RandomStructuresDataGenerator, IsoAtomMaker


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
    data_gen = RandomStructuresDataGenerator(n_struct=3).make(structure=structure, mp_id=test_mpid)

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

    # generate the flow
    flow_iso = IsoAtomMaker().make(species=specie)

    flow_iso = update_user_incar_settings(flow_iso, {"ISMEAR": 0})

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(flow_iso, create_folders=True, ensure_success=True)

    assert (
        responses[flow_iso.output.uuid][1].output.output.energy_per_atom == -0.25638457
    )
