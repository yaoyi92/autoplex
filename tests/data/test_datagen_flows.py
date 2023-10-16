from __future__ import annotations

import inspect
from pymatgen.core.structure import Structure
from autoplex.data.flows import DataGenerator, IsoAtomMaker


def test_data_generation():
    test_structure = Structure(
        lattice=[
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        species=["Mo", "C", "K"],
        coords=[[0.66, 0.66, 0.66], [0.33, 0.33, 0.33], [0, 0, 0]],
    )
    test_mpid = "mp-test"
    test_species = test_structure.types_of_species

    data = DataGenerator().make(structure=test_structure, mp_id=test_mpid)
    isoatom = []
    for species in test_species:
        isoatom.append(IsoAtomMaker().make(species))

    assert (
        len(data.jobs) == 2
    )  # not sure how else checking if job is submitted correctly without actually running VASP jobs
    assert len(isoatom) == 3
    assert [
        items.default
        for items in inspect.signature(DataGenerator).parameters.values()
        if items.name == "sc"
    ][
        0
    ] is False  # important to avoid accidentally way too large workflows


def test_iso_atom_maker(mock_vasp, clean_dir):
    from jobflow import run_locally
    from pymatgen.core import Species
    from atomate2.vasp.powerups import (
        update_user_incar_settings,
    )

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
