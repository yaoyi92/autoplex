from __future__ import annotations

from pymatgen.core.structure import Structure
from atomate2.vasp.powerups import update_user_incar_settings
from autoplex.auto.flows import (
    PhononDFTMLDataGenerationFlow,
    PhononDFTMLBenchmarkFlow,
)


def test_phonon_dft_ml_data_generation_flow(vasp_test_dir, mock_vasp, clean_dir):
    # TODO: this test actually fails but reported passing (something is wrong)

    from jobflow import run_locally

    path_to_struct = vasp_test_dir / "Data_generator" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    flow_data_generation = PhononDFTMLDataGenerationFlow(n_struct=1, min_length=3).make(
        structure=structure, mp_id="mp-22905"
    )

    ref_paths = {
        "tight relax 1": "Data_generator/rand_displacements/",
        "tight relax 2": "Data_generator/rand_displacements/",
        "static": "Data_generator/rand_displacements/",
        "phonon static 1/3": "Data_generator/rand_displacements/",
        "phonon static 2/3": "Data_generator/rand_displacements/",
        "phonon static 3/3": "Data_generator/rand_displacements/",
    }

    fake_run_vasp_kwargs = {
        "check_inputs": ["incar", "kpoints", "potcar"],
        "phonon static 1/3": {"incar_settings": ["NSW", "ISMEAR"]},
        "phonon static 2/3": {"incar_settings": ["NSW", "ISMEAR"]},
        "phonon static 3/3": {"incar_settings": ["NSW", "ISMEAR"]},
    }

    flow_data_generation = update_user_incar_settings(
        flow_data_generation, {"ISMEAR": 0}
    )

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        flow_data_generation, create_folders=True, ensure_success=True
    )
