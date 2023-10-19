from __future__ import annotations

from pymatgen.core.structure import Structure
from autoplex.auto.flows import PhononDFTMLDataGenerationFlow


def test_phonon_dft_ml_data_generation_flow(
    vasp_test_dir, mock_vasp, clean_dir, memory_jobstore
):
    from jobflow import run_locally

    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    flow_data_generation = PhononDFTMLDataGenerationFlow(
        n_struct=3, min_length=10, symprec=1e-2
    ).make(structure=structure, mp_id="mp-22905")

    ref_paths = {
        "tight relax 1": "dft_ml_data_generation/tight_relax_1/",
        "tight relax 2": "dft_ml_data_generation/tight_relax_2/",
        "static": "dft_ml_data_generation/static/",
        "phonon static 1/2": "dft_ml_data_generation/phonon_static_1/",
        "phonon static 2/2": "dft_ml_data_generation/phonon_static_2/",
        "phonon static 1/3": "dft_ml_data_generation/rand_static_1/",
        "phonon static 2/3": "dft_ml_data_generation/rand_static_2/",
        "phonon static 3/3": "dft_ml_data_generation/rand_static_3/",
    }

    fake_run_vasp_kwargs = {
        "tight relax 1": {"incar_settings": ["NSW", "ISMEAR"]},
        "tight relax 2": {"incar_settings": ["NSW", "ISMEAR"]},
        "phonon static 1/2": {"incar_settings": ["NSW"]},
        "phonon static 2/2": {"incar_settings": ["NSW"]},
        "phonon static 1/3": {
            "incar_settings": ["NSW"],
            "check_inputs": ["incar", "kpoints", "potcar"],
        },
        "phonon static 2/3": {
            "incar_settings": ["NSW"],
            "check_inputs": ["incar", "kpoints", "potcar"],
        },
        "phonon static 3/3": {
            "incar_settings": ["NSW"],
            "check_inputs": ["incar", "kpoints", "potcar"],
        },
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        flow_data_generation,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )

    uuids_phonon_calcs = {}
    for k, v in flow_data_generation.output.items():
        if k in ("rand_struc_dir", "phonon_dir"):
            uuids_phonon_calcs[v[0].output.uuid] = k

    paths_to_phonon_calcs = []
    for key in responses.keys():
        if key in uuids_phonon_calcs:
            if uuids_phonon_calcs[key] == "phonon_dir":
                for path in responses[key][1].output.jobdirs.displacements_job_dirs:
                    paths_to_phonon_calcs.append(path)
                    # print(responses[key][1].output.jobdirs.displacements_job_dirs)
            else:
                for output in responses[key][2].output["dirs"]:
                    paths_to_phonon_calcs.append(output.resolve(store=memory_jobstore))
                    # print(output.resolve(store=memory_jobstore))
                    # print(responses[key][2].output['dirs'])

    assert len(paths_to_phonon_calcs) == 5
