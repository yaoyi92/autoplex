from __future__ import annotations

from pymatgen.core.structure import Structure
from autoplex.auto.jobs import get_phonon_ml_calculation_jobs, get_iso_atom
from atomate2.common.schemas.phonons import PhononBSDOSDoc

from jobflow import run_locally


def test_get_phonon_ml_calculation_jobs(test_dir, clean_dir, memory_jobstore):
    potential_file_dir = test_dir / "fitting" / "ref_files" / "gap.xml"
    path_to_struct = test_dir / "fitting" / "ref_files" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    gap_phonon_jobs = get_phonon_ml_calculation_jobs(
        ml_dir=potential_file_dir, structure=structure, min_length=20
    )

    responses = run_locally(
        gap_phonon_jobs, create_folders=True, ensure_success=True, store=memory_jobstore
    )

    ml_phonon_bs_doc = responses[gap_phonon_jobs.output.uuid][2].output.resolve(
        store=memory_jobstore
    )
    assert isinstance(ml_phonon_bs_doc, PhononBSDOSDoc)

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
        "Li-statisoatom": {"incar_settings": ["NSW"], "check_inputs": ["incar", "kpoints"]},
        "Cl-statisoatom": {"incar_settings": ["NSW"], "check_inputs": ["incar", "kpoints"]},
        "C-statisoatom": {"incar_settings": ["NSW"], "check_inputs": ["incar", "kpoints"]},
        "Mo-statisoatom": {"incar_settings": ["NSW"], "check_inputs": ["incar", "kpoints"]},
        "K-statisoatom": {"incar_settings": ["NSW"], "check_inputs": ["incar", "kpoints"]},
        "Si-statisoatom": {"incar_settings": ["NSW"], "check_inputs": ["incar", "kpoints"]},
        "Na-statisoatom": {"incar_settings": ["NSW"], "check_inputs": ["incar", "kpoints"]},
    }

    # automatically use fake VASP and write POTCAR.spec during the test
    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    isolated_atom = get_iso_atom(structure_list)

    # run the flow or job and ensure that it finished running successfully
    responses = run_locally(isolated_atom, create_folders=True, ensure_success=True)

    assert "[Element Li, Element C, Element Mo, Element Na, Element Si, Element Cl, Element K]" == f"{responses[isolated_atom.output.uuid][2].output['species']}"
    assert "Li" and "C" and "Mo" and "Na" and "Si" and "Cl" and "K" in f"{responses[isolated_atom.output.uuid][2].output['species']}"
