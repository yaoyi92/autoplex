from pymatgen.core.structure import Structure
from autoplex.data.phonons.jobs import (
    generate_randomized_structures,
    phonon_maker_random_structures,
)


def test_generate_randomized_structures():
    from jobflow import run_locally

    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    rand_structs_job = generate_randomized_structures(structure=structure, n_struct=10)

    responses = run_locally(rand_structs_job, create_folders=False, ensure_success=True)

    # check if correct number of valid structure objects are generated
    for uuid, response_collection in responses.items():
        for k, response in response_collection.items():
            assert 10 == len(response.output)
            for struct in response.output:
                assert isinstance(struct, Structure)


# def test_phonon_maker_random_structures(
# this function might be redundant ==> rattled structres are only needed to generate more data,
# we don't need to calculate the phonon structure of them
#     vasp_test_dir, clean_dir, memory_jobstore, mock_vasp
# ):
#     from pymatgen.core import Structure
#     from atomate2.common.schemas.phonons import PhononBSDOSDoc
#     from atomate2.common.jobs.phonons import PhononDisplacementMaker
#     from jobflow import run_locally
#
#     ref_paths = {
#         "static": "dft_ml_data_generation/static/",
#         "phonon static 1/2": "dft_ml_data_generation/phonon_static_1/",
#         "phonon static 2/2": "dft_ml_data_generation/phonon_static_2/",
#     }
#
#     fake_run_vasp_kwargs = {
#         "static": {
#             "incar_settings": ["NSW"],
#             "check_inputs": ["incar", "kpoints", "potcar"],
#         },
#         "phonon static 1/2": {
#             "incar_settings": ["NSW"],
#             "check_inputs": ["incar", "kpoints", "potcar"],
#         },
#         "phonon static 2/2": {
#             "incar_settings": ["NSW"],
#             "check_inputs": ["incar", "kpoints", "potcar"],
#         },
#     }
#
#     structure1 = Structure.from_file(
#         vasp_test_dir / "dft_ml_data_generation" / "rand_static_1" / "inputs" / "POSCAR"
#     )
#     structure2 = Structure.from_file(
#         vasp_test_dir / "dft_ml_data_generation" / "rand_static_2" / "inputs" / "POSCAR"
#     )
#     structure3 = Structure.from_file(
#         vasp_test_dir / "dft_ml_data_generation" / "rand_static_3" / "inputs" / "POSCAR"
#     )
#
#     rattled_structure_list = [structure1, structure2, structure3]
#
#     random_phonon_job = phonon_maker_random_structures(
#         rattled_structures=rattled_structure_list,
#         symprec=0.01,
#         displacements=[0.01],
#         phonon_displacement_maker=PhononDisplacementMaker(),
#     )
#
#     # automatically use fake VASP and write POTCAR.spec during the test
#     mock_vasp(ref_paths, fake_run_vasp_kwargs)
#
#     responses = run_locally(
#         random_phonon_job,
#         create_folders=True,
#         ensure_success=True,
#         store=memory_jobstore,
#     )
#
#     for output in responses[random_phonon_job.output.uuid][2].output:
#         assert isinstance(output.resolve(store=memory_jobstore), PhononBSDOSDoc)

# TODO add a unit test to check if the supercell rattled structures have been constructed properly
