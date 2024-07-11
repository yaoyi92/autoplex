from __future__ import annotations

import os
from unittest import mock
from pymatgen.core.structure import Structure
from autoplex.auto.phonons.jobs import (
    get_iso_atom,
    dft_phonopy_gen_data,
)
from atomate2.common.schemas.phonons import PhononBSDOSDoc
from autoplex.data.phonons.flows import TightDFTStaticMaker
from jobflow import run_locally

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1


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
    dft_phonon_workflow = dft_phonopy_gen_data(structure, [0.01], 0.1, TightDFTStaticMaker(), 10)

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
        dft_phonon_workflow.output.resolve(store=memory_jobstore)["data"]["001"],
        PhononBSDOSDoc,
    )


def test_phonopy_supercell_settings(memory_jobstore, clean_dir):
    structure = Structure(  # mp-1203790
        lattice=[[-5.183318, -8.977762, 0.000000], [-5.183315, 8.977761, -0.000000], [0.000000, 0.000000, -16.970272]],
        species=["Si"] * 68,
        coords=[[0.95006076, 0.66941597, 0.56452266],
                [0.71935521, 0.04993924, 0.56452366],
                [0.33058503, 0.28064479, 0.56452266],
                [0.71935321, 0.66941397, 0.56452366],
                [0.33058603, 0.04993924, 0.56452366],
                [0.95006076, 0.28064579, 0.56452366],
                [0.04993924, 0.33058403, 0.43547534],
                [0.28064479, 0.95006076, 0.43547534],
                [0.66941597, 0.71935521, 0.43547534],
                [0.28064679, 0.33058603, 0.43547534],
                [0.66941497, 0.95006076, 0.43547534],
                [0.04993924, 0.71935421, 0.43547534],
                [0.04994024, 0.33058403, 0.06452466],
                [0.28064479, 0.95005976, 0.06452466],
                [0.66941497, 0.71935521, 0.06452466],
                [0.28064679, 0.33058603, 0.06452466],
                [0.66941397, 0.95005976, 0.06452466],
                [0.04994024, 0.71935421, 0.06452466],
                [0.95005976, 0.66941597, 0.93547734],
                [0.71935521, 0.04994024, 0.93547634],
                [0.33058503, 0.28064479, 0.93547634],
                [0.71935321, 0.66941397, 0.93547634],
                [0.33058603, 0.04994024, 0.93547634],
                [0.95005976, 0.28064579, 0.93547634],
                [0.91572454, 0.45786277, 0.86478307],
                [0.54213923, 0.08427546, 0.86478307],
                [0.54213723, 0.45786077, 0.86478307],
                [0.08427546, 0.54213723, 0.13521793],
                [0.45786077, 0.91572454, 0.13521793],
                [0.45786277, 0.54213923, 0.13521793],
                [0.08427646, 0.54213723, 0.36478107],
                [0.45786077, 0.91572354, 0.36478107],
                [0.45786277, 0.54214023, 0.36478107],
                [0.91572354, 0.45786277, 0.63521793],
                [0.54213923, 0.08427646, 0.63521793],
                [0.54213723, 0.45785977, 0.63521793],
                [0.87645037, 0.12354863, 0.36130919],
                [0.24709825, 0.12354963, 0.36130919],
                [0.87645037, 0.75290175, 0.36130919],
                [0.12354963, 0.87645137, 0.63868881],
                [0.75290175, 0.87645037, 0.63868881],
                [0.12354963, 0.24709825, 0.63868881],
                [0.12354963, 0.87645137, 0.86131119],
                [0.75290175, 0.87645037, 0.86131119],
                [0.12354963, 0.24709825, 0.86131119],
                [0.87645037, 0.12354863, 0.13869081],
                [0.24709825, 0.12354963, 0.13868981],
                [0.87645037, 0.75290175, 0.13868981],
                [0.666667, 0.333333, 0.8196504],
                [0.333333, 0.666667, 0.1803516],
                [0.333333, 0.666667, 0.3196484],
                [0.666667, 0.333333, 0.6803506],
                [-0., -0., 0.81830178],
                [-0., -0., 0.18169922],
                [-0., -0., 0.31829978],
                [-0., -0., 0.68169922],
                [0.93042857, 0.46521428, 0.25],
                [0.53478672, 0.06957143, 0.25],
                [0.53478472, 0.46521328, 0.25],
                [0.06957243, 0.53478572, 0.75],
                [0.46521228, 0.93042757, 0.75],
                [0.46521428, 0.53478772, 0.75],
                [0.79765372, 0.20234828, 0.25],
                [0.40469355, 0.20234628, 0.25],
                [0.79765172, 0.59530645, 0.25],
                [0.20234628, 0.79765172, 0.75],
                [0.59530545, 0.79765272, 0.75],
                [0.20234928, 0.40469455, 0.75]],
    )
    dft_phonon = dft_phonopy_gen_data(structure, [0.01], 0.1, TightDFTStaticMaker(), 18)

    run_locally(
        dft_phonon,
        create_folders=False,
        ensure_success=False,
        # this unit test is just to check if phonon_displacement_maker settings are successfully adapted
    )

    print("Finish unit test")
