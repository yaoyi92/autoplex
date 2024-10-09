from __future__ import annotations

import os
from unittest import mock
from jobflow import Flow
from pymatgen.core.structure import Structure
from autoplex.auto.phonons.jobs import (
    get_iso_atom,
    dft_phonopy_gen_data,
    complete_benchmark
)

from atomate2.common.schemas.phonons import PhononBSDOSDoc
from autoplex.data.phonons.flows import TightDFTStaticMaker
from jobflow import run_locally
from tests.conftest import memory_jobstore
from pytest import approx

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1


def test_complete_benchmark(clean_dir, test_dir, memory_jobstore):
    from monty.serialization import loadfn
    from atomate2.common.schemas.phonons import PhononBSDOSDoc
    from autoplex.fitting.common.flows import MLIPFitMaker
    database_dir = test_dir / "fitting/rss_training_dataset/"
    jobs = []
    gapfit = MLIPFitMaker().make(
        auto_delta=False,
        glue_xml=False,
        twob={"delta": 2.0, "cutoff": 4},
        threeb={"n_sparse": 10},
        preprocessing_data=False,
        database_dir=database_dir,
        separated=True
    )
    dft_data = loadfn(test_dir / "benchmark" / "phonon_doc_si.json")
    dft_doc: PhononBSDOSDoc = dft_data["output"]
    structure = dft_doc.structure

    jobs.append(gapfit)

    bm = complete_benchmark(ibenchmark_structure=0, benchmark_structure=structure, mp_ids=["mp-82"],
                            benchmark_mp_ids=["mp-82"], ml_path=gapfit.output["mlip_path"], ml_model="GAP",
                            dft_references=[dft_doc], add_dft_phonon_struct=False, fit_input=None, symprec=1e-1,
                            phonon_displacement_maker=None, supercell_settings={"min_length": 8})
    jobs.append(bm)

    response = run_locally(Flow(jobs), store=memory_jobstore)
    output = response[bm.output.uuid][1].output[0].resolve(store=memory_jobstore)
    assert output["benchmark_phonon_rmse"] == approx(1.0, abs=0.5)
    assert output["dft_imaginary_modes"] == False
    assert output["ml_imaginary_modes"] == False


def test_get_iso_atom(vasp_test_dir, mock_vasp, clean_dir, memory_jobstore):
    structure_list = [
        Structure(
            lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
            species=["Si", "Si"],
            coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
        ),
        Structure(
            lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
            species=["Mo", "C", "K"],
            coords=[[0, 0, 0], [0.25, 0.25, 0.25], [0.55, 0.55, 0.55]],
        ),
        Structure(
            lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
            species=["Mo", "K"],
            coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
        ),
        Structure(
            lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
            species=["Li", "Na", "K"],
            coords=[[0, 0, 0], [0.25, 0.25, 0.25], [0.55, 0.55, 0.55]],
        ),
        Structure(
            lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
            species=["Li", "Li"],
            coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
        ),
        Structure(
            lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
            species=["Li", "Cl"],
            coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
        ),
    ]

    ref_paths = {
        "Li-statisoatom": "Li_iso_atoms/Li-statisoatom/",
        "Cl-statisoatom": "Cl_iso_atoms/Cl-statisoatom/",
        "C-statisoatom": "Cl_iso_atoms/Cl-statisoatom/",
        "Mo-statisoatom": "Cl_iso_atoms/Cl-statisoatom/",
        "K-statisoatom": "Cl_iso_atoms/Cl-statisoatom/",
        "Si-statisoatom": "Cl_iso_atoms/Cl-statisoatom/",
        "Na-statisoatom": "Cl_iso_atoms/Cl-statisoatom/",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_vasp_kwargs = {
        "Li-statisoatom": {
            "incar_settings": ["NSW"],
            "check_inputs": ["incar", "kpoints"],
        },
        "Cl-statisoatom": {
            "incar_settings": ["NSW"],
            "check_inputs": ["incar", "kpoints"],
        },
        "C-statisoatom": {
            "incar_settings": ["NSW"],
            "check_inputs": ["incar", "kpoints"],
        },
        "Mo-statisoatom": {
            "incar_settings": ["NSW"],
            "check_inputs": ["incar", "kpoints"],
        },
        "K-statisoatom": {
            "incar_settings": ["NSW"],
            "check_inputs": ["incar", "kpoints"],
        },
        "Si-statisoatom": {
            "incar_settings": ["NSW"],
            "check_inputs": ["incar", "kpoints"],
        },
        "Na-statisoatom": {
            "incar_settings": ["NSW"],
            "check_inputs": ["incar", "kpoints"],
        },
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    isolated_atom = get_iso_atom(structure_list)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(isolated_atom, create_folders=True, ensure_success=True)

    assert (
            "[Element Li, Element C, Element Mo, Element Na, Element Si, Element Cl, Element K]"
            == f"{responses[isolated_atom.output.uuid][2].output['species']}"
    )
    assert (
            "Li"
            and "C"
            and "Mo"
            and "Na"
            and "Si"
            and "Cl"
            and "K" in f"{responses[isolated_atom.output.uuid][2].output['species']}"
    )


def test_dft_task_doc(vasp_test_dir, mock_vasp, test_dir, memory_jobstore, clean_dir):
    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)
    dft_phonon_workflow = dft_phonopy_gen_data(structure=structure, displacements=[0.01], symprec=0.1,
                                               phonon_displacement_maker=TightDFTStaticMaker(),
                                               supercell_settings={"min_length": 10, "min_atoms": 20})

    ref_paths = {
        "tight relax 1": "dft_ml_data_generation/tight_relax_1/",
        "tight relax 2": "dft_ml_data_generation/tight_relax_2/",
        "static": "dft_ml_data_generation/static/",
        "dft static 1/2": "dft_ml_data_generation/phonon_static_1/",
        "dft static 2/2": "dft_ml_data_generation/phonon_static_2/",
        "dft static 1/3": "dft_ml_data_generation/rand_static_1/",
        "dft static 2/3": "dft_ml_data_generation/rand_static_2/",
        "dft static 3/3": "dft_ml_data_generation/rand_static_3/",
    }

    fake_run_vasp_kwargs = {
        "tight relax 1": {"incar_settings": ["NSW", "ISMEAR"]},
        "tight relax 2": {"incar_settings": ["NSW", "ISMEAR"]},
        "dft static 1/2": {"incar_settings": ["NSW"]},
        "dft static 2/2": {"incar_settings": ["NSW"]},
        "dft static 1/3": {
            "incar_settings": ["NSW"],
            "check_inputs": ["incar", "kpoints", "potcar"],
        },
        "dft static 2/3": {
            "incar_settings": ["NSW"],
            "check_inputs": ["incar", "kpoints", "potcar"],
        },
        "dft static 3/3": {
            "incar_settings": ["NSW"],
            "check_inputs": ["incar", "kpoints", "potcar"],
        },
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(
        dft_phonon_workflow,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore,
    )

    # check for DFT phonon doc
    assert isinstance(
        dft_phonon_workflow.output.resolve(store=memory_jobstore)["phonon_data"]["001"],
        PhononBSDOSDoc,
    )
