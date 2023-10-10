from pymatgen.core.structure import Structure
from autoplex.data.jobs import generate_randomized_structures


def test_generate_randomized_structures():
    from jobflow import run_locally

    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    rand_structs_job = generate_randomized_structures(structure=structure, n_struc=10)

    responses = run_locally(rand_structs_job, create_folders=False, ensure_success=True)

    # check if correct number of valid structure objects are generated
    for uuid, response_collection in responses.items():
        for k, response in response_collection.items():
            assert 10 == len(response.output)
            for struct in response.output:
                assert isinstance(struct, Structure)
