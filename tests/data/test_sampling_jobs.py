from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from autoplex.data.common.jobs import Sampling
from jobflow import run_locally

def test_sampling_cur():
    TEST_FILEPATH = "/u/vld/iclb0745/autoplex_code_test/rss_merge/autoplex/tests/test_data/data/rss.extxyz"
    atoms = read(TEST_FILEPATH, index=':')
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    selection_method = 'cur'
    num_of_selection = 5

    print(len(structures))
    job = Sampling(
        selection_method=selection_method,
        num_of_selection=num_of_selection,
        structure=structures
    )
    
    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True
    )

    selected_atoms = response[job.uuid]["output"]

    assert len(selected_atoms) == num_of_selection

    for atom in selected_atoms:
        assert isinstance(atom, type(structures[0]))

