from __future__ import annotations
import os
from unittest import mock
from atomate2.common.schemas.phonons import PhononBSDOSDoc
from pymatgen.core.structure import Structure
from autoplex.auto.flows import (
    CompleteDFTvsMLBenchmarkWorkflow,
    DFTDataGenerationFlow,
)

mock.patch.dict(os.environ, {"OMP_NUM_THREADS": 1, "OPENBLAS_OMP_THREADS": 2})


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
        benchmark_mp_id="mp-22905",
        benchmark_structures=structure,
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

    ml_task_doc = responses[complete_workflow.jobs[4].output.uuid][2].output.resolve(
        store=memory_jobstore
    )

    assert isinstance(ml_task_doc, PhononBSDOSDoc)

    assert responses[complete_workflow.jobs[6].output.uuid][1].output == pytest.approx(
        0.5716963823412201, abs=0.02
    )


def test_add_data_to_dataset_workflow(
    vasp_test_dir, mock_vasp, test_dir, memory_jobstore, clean_dir
):
    import pytest
    from monty.serialization import loadfn
    from jobflow import run_locally
    from atomate2.common.jobs.phonons import PhononDisplacementMaker

    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)
    dft_data = loadfn(test_dir / "benchmark" / "PhononBSDOSDoc_LiCl.json")
    dft_reference: PhononBSDOSDoc = dft_data["output"]

    add_data_workflow = CompleteDFTvsMLBenchmarkWorkflow(
        n_struct=3,
        symprec=1e-2,
        min_length=8,
        displacements=[0.01],
        phonon_displacement_maker=PhononDisplacementMaker(),
    ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids="mp-22905",
        benchmark_structures=structure,
        xyz_file=test_dir / "fitting" / "ref_files" / "trainGAP.xyz",
        dft_references=None,
    )

    add_data_workflow_with_dft_reference = CompleteDFTvsMLBenchmarkWorkflow(
        n_struct=3,
        symprec=1e-2,
        min_length=8,
        displacements=[0.01],
        add_dft_phonon_struct=False,
        phonon_displacement_maker=PhononDisplacementMaker(),
    ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids="mp-22905",
        benchmark_structures=structure,
        xyz_file=test_dir / "fitting" / "ref_files" / "trainGAP.xyz",
        dft_references=dft_reference,
    )

    add_data_workflow_add_phonon_false = CompleteDFTvsMLBenchmarkWorkflow(
        n_struct=3,
        symprec=1e-2,
        min_length=8,
        displacements=[0.01],
        add_dft_phonon_struct=False,
        phonon_displacement_maker=PhononDisplacementMaker(),
    ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids="mp-22905",
        benchmark_structures=structure,
        xyz_file=test_dir / "fitting" / "ref_files" / "trainGAP.xyz",
        dft_references=None,
    )

    add_data_workflow_add_random_false = CompleteDFTvsMLBenchmarkWorkflow(
        n_struct=3,
        symprec=1e-2,
        min_length=8,
        displacements=[0.01],
        add_dft_random_struct=False,
        phonon_displacement_maker=PhononDisplacementMaker(),
    ).make(
        structure_list=[structure],
        mp_ids=["test"],
        benchmark_mp_ids="mp-22905",
        benchmark_structures=structure,
        xyz_file=test_dir / "fitting" / "ref_files" / "trainGAP.xyz",
        dft_references=None,
    )

    add_data_workflow_with_same_mpid = CompleteDFTvsMLBenchmarkWorkflow(
        n_struct=3,
        symprec=1e-2,
        min_length=8,
        displacements=[0.01],
        phonon_displacement_maker=PhononDisplacementMaker(),
    ).make(
        structure_list=[structure],
        mp_ids=["mp-22905"],
        benchmark_mp_ids="mp-22905",
        benchmark_structures=structure,
        xyz_file=test_dir / "fitting" / "ref_files" / "trainGAP.xyz",
        dft_references=None,
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

    responses_wdft = run_locally(
        add_data_workflow_with_dft_reference,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )

    assert responses[add_data_workflow.jobs[6].output.uuid][1].output == pytest.approx(
        0.5716963823412201, abs=0.02
    )
    for job, uuid in add_data_workflow.iterflow():
        if "dft_phonopy_gen_data" in job.name:
            assert True
    for job, uuid in add_data_workflow_with_dft_reference.iterflow():
        assert job.name != "dft_phonopy_gen_data"
    for job, uuid in add_data_workflow_add_phonon_false.iterflow():
        assert job.name != "dft_phonopy_gen_data"
    for job, uuid in add_data_workflow_add_random_false.iterflow():
        assert job.name != "dft_random_gen_data"
    for job, uuid in add_data_workflow_with_dft_reference.iterflow():
        assert job.name != "tight relax 1"
    for job, uuid in add_data_workflow_with_same_mpid.iterflow():
        assert job.name != "tight relax 1"


def test_phonon_dft_ml_data_generation_flow(
    vasp_test_dir, mock_vasp, clean_dir, memory_jobstore
):
    from jobflow import run_locally

    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    flow_data_generation = DFTDataGenerationFlow(
        n_struct=3, min_length=10, symprec=1e-2
    ).make(structure=structure, benchmark_mp_ids="mp-22905")

    flow_data_generation_without_rattled_structures = DFTDataGenerationFlow(
        n_struct=0, min_length=10, symprec=1e-2
    ).make(structure=structure, benchmark_mp_ids="mp-22905")

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

    responses_worattled = run_locally(
        flow_data_generation_without_rattled_structures,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )

    uuids_phonon_calcs = {}
    for k, v in flow_data_generation.output.items():
        if k in ("rand_struc_dir", "phonon_dir"):
            uuids_phonon_calcs[v[0].output.uuid] = k

    uuids_phonon_calcs_worattled = {}
    for k, v in flow_data_generation_without_rattled_structures.output.items():
        if k in ("rand_struc_dir", "phonon_dir"):
            uuids_phonon_calcs_worattled[v[0].output.uuid] = k
            assert k != "rand_struc_dir"

    paths_to_phonon_calcs = []
    paths_to_rand_calcs = []
    for key in responses.keys():
        if key in uuids_phonon_calcs:
            if uuids_phonon_calcs[key] == "phonon_dir":
                for output in responses[key][2].output["dirs"]:
                    for item in output.resolve(store=memory_jobstore):
                        paths_to_phonon_calcs.append(item)
            if uuids_phonon_calcs[key] == "rand_struc_dir":
                for output in responses[key][2].output:
                    for item in output.resolve(store=memory_jobstore):
                        paths_to_rand_calcs.append(item)

    assert len(paths_to_phonon_calcs) + len(paths_to_rand_calcs) == 5

    paths_to_phonon_calcs_worattled = []
    paths_to_rand_calcs_worattled = []
    for key in responses_worattled.keys():
        if key in uuids_phonon_calcs_worattled:
            if uuids_phonon_calcs_worattled[key] == "phonon_dir":
                for output in responses_worattled[key][2].output["dirs"]:
                    for item in output.resolve(store=memory_jobstore):
                        paths_to_phonon_calcs_worattled.append(item)
            if uuids_phonon_calcs_worattled[key] == "rand_struc_dir":
                for output in responses_worattled[key][2].output:
                    for item in output.resolve(store=memory_jobstore):
                        paths_to_rand_calcs_worattled.append(item)

    assert (
        len(paths_to_phonon_calcs_worattled) + len(paths_to_rand_calcs_worattled) == 2
    )
