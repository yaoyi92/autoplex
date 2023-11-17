from __future__ import annotations

from pymatgen.core.structure import Structure
from autoplex.auto.flows import (
    CompleteDFTvsMLBenchmarkWorkflow,
    AddDataToDataset,
    DFTDataGenerationFlow,
)


def test_complete_dft_vs_ml_benchmark_workflow(
    vasp_test_dir, mock_vasp, test_dir, memory_jobstore, clean_dir
):
    import pytest
    from jobflow import run_locally
    from atomate2.common.jobs.phonons import PhononDisplacementMaker
    from atomate2.common.schemas.phonons import PhononBSDOSDoc

    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    complete_workflow = CompleteDFTvsMLBenchmarkWorkflow(
        n_struct=3, symprec=1e-2, min_length=8, displacements=[0.01]
    ).make(
        structure_list=[structure],
        mp_ids=["test"],
        mp_id="mp-22905",
        benchmark_structure=structure,
        phonon_displacement_maker=PhononDisplacementMaker(),
    )

    ref_paths = {
        "tight relax 1": "dft_ml_data_generation/tight_relax_1/",
        "tight relax 2": "dft_ml_data_generation/tight_relax_2/",
        "static": "dft_ml_data_generation/static/",
        "Cl-statisoatom": "Cl_iso_atoms/Cl-statisoatom/",
        "Li-statisoatom": "Li_iso_atoms/Li-statisoatom/",
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
        complete_workflow,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )
    # check for DFT phonon doc
    for k, v in complete_workflow.jobs[2].output.items():
        if k == "phonon_data":
            assert isinstance(responses[v[0].uuid][1].output, PhononBSDOSDoc)

    # check for ML phonon doc
    ml_task_doc = responses[complete_workflow.jobs[4].output.uuid][2].output.resolve(
        store=memory_jobstore
    )
    assert isinstance(ml_task_doc, PhononBSDOSDoc)
    assert responses[complete_workflow.jobs[5].output.uuid][1].output == pytest.approx(
        80.32601884386796, abs=0.1
    )

def test_add_data_to_dataset_workflow(
    vasp_test_dir, mock_vasp, test_dir, memory_jobstore, clean_dir
):  # TODO: add test cases for add_dft_random_struct=False and add_dft_phonon_struct=False
    import pytest
    from jobflow import run_locally
    from atomate2.common.jobs.phonons import PhononDisplacementMaker
    from atomate2.common.schemas.phonons import PhononBSDOSDoc

    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    add_data_workflow = AddDataToDataset(
        n_struct=3, symprec=1e-2, min_length=8, displacements=[0.01], phonon_displacement_maker=PhononDisplacementMaker()
    ).make(
        structure_list=[structure],
        mp_ids=["test"],
        mp_id="mp-22905",
        benchmark_structure=structure,
        xyz_file= test_dir / "fitting" / "ref_files" / "trainGAP.xyz"
    )

    ref_paths = {
        "tight relax 1": "dft_ml_data_generation/tight_relax_1/",
        "tight relax 2": "dft_ml_data_generation/tight_relax_2/",
        "static": "dft_ml_data_generation/static/",
        "Cl-statisoatom": "Cl_iso_atoms/Cl-statisoatom/",
        "Li-statisoatom": "Li_iso_atoms/Li-statisoatom/",
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
        add_data_workflow,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )
    # check for DFT phonon doc
    for k, v in add_data_workflow.jobs[2].output.items():
        if k == "phonon_data":
            assert isinstance(responses[v[0].uuid][1].output, PhononBSDOSDoc)

    # check for ML phonon doc
    ml_task_doc = responses[add_data_workflow.jobs[4].output.uuid][2].output.resolve(
        store=memory_jobstore
    )
    assert isinstance(ml_task_doc, PhononBSDOSDoc)
    assert responses[add_data_workflow.jobs[5].output.uuid][1].output == pytest.approx(
        80.32601884386796, abs=0.1
    )

def test_phonon_dft_ml_data_generation_flow(
    vasp_test_dir, mock_vasp, clean_dir, memory_jobstore
):
    from jobflow import run_locally

    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    flow_data_generation = DFTDataGenerationFlow(
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
