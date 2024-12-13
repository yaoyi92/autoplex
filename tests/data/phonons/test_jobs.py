import pytest
from jobflow import run_locally
from pymatgen.core.structure import Structure
from autoplex.data.phonons.jobs import reduce_supercell_size_job


@pytest.fixture(scope="class")
def mp_1200830():
    return Structure.from_dict(  # mp-1200830
        {'@module': 'pymatgen.core.structure', '@class': 'Structure', 'charge': 0, 'lattice': {
            'matrix': [[-6.2141887, -6.2141887, 6.2141887], [-6.2141887, 6.2141887, -6.2141887],
                       [6.2141887, -6.2141887, -6.2141887]], 'pbc': (True, True, True), 'a': 10.763290556220392,
            'b': 10.763290556220392, 'c': 10.763290556220392, 'alpha': 109.47122063449069, 'beta': 109.47122063449069,
            'gamma': 109.47122063449069, 'volume': 959.8719531108837}, 'properties': {}, 'sites': [
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.78659388, 0.39175378, 0.13262899],
             'xyz': [-6.498293142493029, -3.277792458677283, 1.629429316776457], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.3948401, 0.74087421, 0.60824622],
                              'xyz': [-3.2777862444885835, -1.6294355309651567, -5.930078043318272],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.65396589, 0.86737101, 0.25912579],
             'xyz': [-7.843618016776459, -0.28411687087042936, -2.9363962413227167], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.65396589, 0.25912579, 0.86737101],
                              'xyz': [-0.28411687087042914, -7.843618016776459, -2.9363962413227167],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.78659388, 0.13262899, 0.39175378],
             'xyz': [-3.2777924586772835, -6.498293142493028, 1.6294293167764566], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.3948401, 0.60824622, 0.74087421],
                              'xyz': [-1.6294355309651563, -3.2777862444885835, -5.930078043318272],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.25912579, 0.65396589, 0.86737101],
             'xyz': [-0.28411687087042914, -2.9363962413227167, -7.843618016776459], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.13262899, 0.78659388, 0.39175378],
                              'xyz': [-3.2777924586772835, 1.6294293167764566, -6.498293142493028],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.60824622, 0.3948401, 0.74087421],
             'xyz': [-1.6294355309651563, -5.930078043318272, -3.2777862444885835], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.39175378, 0.78659388, 0.13262899],
                              'xyz': [-6.498293142493029, 1.629429316776457, -3.277792458677283],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.74087421, 0.3948401, 0.60824622],
             'xyz': [-3.2777862444885835, -5.930078043318272, -1.6294355309651567], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.86737101, 0.65396589, 0.25912579],
                              'xyz': [-7.843618016776459, -2.9363962413227167, -0.28411687087042936],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.74087421, 0.60824622, 0.3948401],
             'xyz': [-5.930078043318271, -3.2777862444885835, -1.6294355309651567], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.86737101, 0.25912579, 0.65396589],
                              'xyz': [-2.9363962413227167, -7.843618016776459, -0.28411687087042914],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.39175378, 0.13262899, 0.78659388],
             'xyz': [1.629429316776457, -6.498293142493029, -3.277792458677283], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.60824622, 0.74087421, 0.3948401],
                              'xyz': [-5.930078043318271, -1.6294355309651567, -3.2777862444885835],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.25912579, 0.86737101, 0.65396589],
             'xyz': [-2.9363962413227167, -0.28411687087042914, -7.843618016776459], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.13262899, 0.39175378, 0.78659388],
                              'xyz': [1.629429316776457, -3.277792458677283, -6.498293142493029],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.21340612, 0.34603411, 0.6051599],
             'xyz': [0.28411065668172863, -2.936402455511417, -4.584753169034843], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.6051599, 0.21340612, 0.34603411],
                              'xyz': [-2.936402455511417, -4.584753169034843, 0.28411065668172863],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.34603411, 0.6051599, 0.21340612],
             'xyz': [-4.584753169034843, 0.28411065668172886, -2.936402455511417], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.34603411, 0.21340612, 0.6051599],
                              'xyz': [0.28411065668172863, -4.584753169034843, -2.936402455511417],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.21340612, 0.6051599, 0.34603411],
             'xyz': [-2.936402455511417, 0.28411065668172863, -4.584753169034843], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.6051599, 0.34603411, 0.21340612],
                              'xyz': [-4.584753169034843, -2.936402455511417, 0.28411065668172886],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [-0.0, -0.0, 0.21736945],
             'xyz': [1.350774779915215, -1.350774779915215, -1.350774779915215], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [-0.0, 0.21736945, -0.0],
                              'xyz': [-1.350774779915215, 1.350774779915215, -1.350774779915215],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.78263055, 0.78263055, 0.78263055],
             'xyz': [-4.863413920084786, -4.863413920084786, -4.863413920084786], 'properties': {'magmom': -0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.21736945, 0.0, 0.0],
                              'xyz': [-1.350774779915215, -1.350774779915215, 1.350774779915215],
                              'properties': {'magmom': -0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.0, 0.0, -0.0], 'xyz': [0.0, 0.0, 0.0],
             'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [-0.0, 0.35932015, 0.22356876],
             'xyz': [-0.8435847537472929, 0.8435847537472929, -3.6221816778773173], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.64067985, 0.86424861, 0.64067985],
                              'xyz': [-5.370603946252707, -2.592007022122684, -5.370603946252707],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.77643124, 0.77643124, 0.13575139],
             'xyz': [-8.806195722122682, -0.843584753747293, -0.843584753747293], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.77643124, 0.13575139, 0.77643124],
                              'xyz': [-0.8435847537472929, -8.806195722122684, -0.8435847537472929],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [-0.0, 0.22356876, 0.35932015],
             'xyz': [0.8435847537472929, -0.8435847537472929, -3.6221816778773173], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.64067985, 0.64067985, 0.86424861],
                              'xyz': [-2.592007022122684, -5.370603946252707, -5.370603946252707],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.13575139, 0.77643124, 0.77643124],
             'xyz': [-0.8435847537472929, -0.8435847537472929, -8.806195722122684], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.22356876, -0.0, 0.35932015],
                              'xyz': [0.8435847537472929, -3.6221816778773173, -0.8435847537472929],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.35932015, -0.0, 0.22356876],
             'xyz': [-0.8435847537472929, -3.6221816778773173, 0.8435847537472929], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.86424861, 0.64067985, 0.64067985],
                              'xyz': [-5.370603946252707, -5.370603946252707, -2.592007022122684],
                              'properties': {'magmom': 0.0}, 'label': 'Si'},
            {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.35932015, 0.22356876, -0.0],
             'xyz': [-3.6221816778773173, -0.8435847537472929, 0.8435847537472929], 'properties': {'magmom': 0.0},
             'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.22356876, 0.35932015, -0.0],
                              'xyz': [-3.6221816778773173, 0.8435847537472929, -0.8435847537472929],
                              'properties': {'magmom': 0.0}, 'label': 'Si'}]})


def test_supercell_check(mp_1200830, memory_jobstore):
    expected_matrix = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]

    new_matrix_job = reduce_supercell_size_job(
        min_length=18,
        max_length=25,
        min_atoms=50,
        max_atoms=500,
        fallback_min_length=12,
        structure=mp_1200830,
        step_size=1.0
    )

    run_locally(new_matrix_job, create_folders=False, ensure_success=False, store=memory_jobstore)

    assert new_matrix_job.output.resolve(memory_jobstore) == expected_matrix