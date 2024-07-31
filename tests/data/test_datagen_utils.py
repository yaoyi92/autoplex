from __future__ import annotations

import os
import matplotlib.pyplot as plt
from pymatgen.core.structure import Structure
from autoplex.data.common.utils import (
    energy_plot,
    force_plot,
    plot_energy_forces,
    filter_outlier_energy,
    filter_outlier_forces,
    generate_supercell_matrix,
)
from autoplex.data.phonons.utils import update_phonon_displacement_maker
from atomate2.common.jobs.phonons import get_supercell_size

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1

fig, ax_list = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(15, 20)

def test_energy_forces(clean_dir, test_dir):
    parent_dir = os.getcwd()
    os.chdir(test_dir / "data" / "ref_data")
    energy_forces = plot_energy_forces(
        title="regularization 0.1",
        energy_limit=0.0005,
        force_limit=0.15,
        train_name='train_Si.extxyz',
        test_name='test_Si.extxyz'
    )

    assert os.path.isfile("energy_forces_Si.pdf")
    assert os.path.isfile("energy_forces_Si.png")

    for file_name in os.listdir(os.getcwd()):
        if file_name not in ['train_Si.extxyz', 'test_Si.extxyz', 'quip_train_Si.extxyz', 'quip_test_Si.extxyz']:
            os.remove(os.path.join(os.getcwd(), file_name))

    os.chdir(parent_dir)

def test_energy_plot(test_dir, clean_dir):
    parent_dir = os.getcwd()
    os.chdir(test_dir / "data" / "ref_data")
    energy = energy_plot("train_Si.extxyz", "quip_train_Si.extxyz", ax_list, "Energy on training data")
    plt.savefig("test_energy.png")
    assert os.path.isfile("test_energy.png")

def test_force_plot(test_dir, clean_dir):
    parent_dir = os.getcwd()
    os.chdir(test_dir / "data" / "ref_data")
    force = force_plot("train_Si.extxyz", "quip_train_Si.extxyz", ax_list, "Si", "Force on training data - Si",)
    plt.savefig("test_force.png")
    assert os.path.isfile("test_force.png")

    for file_name in os.listdir(os.getcwd()):
        if file_name not in ['train_Si.extxyz', 'test_Si.extxyz', 'quip_train_Si.extxyz', 'quip_test_Si.extxyz']:
            os.remove(os.path.join(os.getcwd(), file_name))

    os.chdir(parent_dir)

def test_filter_outliers(test_dir, clean_dir):
    parent_dir = os.getcwd()
    os.chdir(test_dir / "data" / "ref_data")
    filter_energy = filter_outlier_energy("train_Si.extxyz", "quip_train_Si.extxyz", 0.005)
    filter_forces = filter_outlier_forces("train_Si.extxyz", "quip_train_Si.extxyz", "Si", 0.1)

    assert os.path.isfile("filtered_in_energy_Si.extxyz")
    assert os.path.isfile("filtered_out_energy_Si.extxyz")
    assert os.path.isfile("outliers_energy_Si.extxyz")
    assert os.path.isfile("filtered_in_force_Si.extxyz")
    assert os.path.isfile("filtered_out_force_Si.extxyz")
    assert os.path.isfile("outliers_force_Si.extxyz")

    for file_name in os.listdir(os.getcwd()):
        if file_name not in ['train_Si.extxyz', 'test_Si.extxyz', 'quip_train_Si.extxyz', 'quip_test_Si.extxyz']:
            os.remove(os.path.join(os.getcwd(), file_name))

    os.chdir(parent_dir)

def test_supercell_check():
    import warnings
    structure = Structure.from_dict(  # mp-1200830
        {'@module': 'pymatgen.core.structure', '@class': 'Structure', 'charge': 0, 'lattice': {'matrix': [[-6.2141887, -6.2141887, 6.2141887], [-6.2141887, 6.2141887, -6.2141887], [6.2141887, -6.2141887, -6.2141887]], 'pbc': (True, True, True), 'a': 10.763290556220392, 'b': 10.763290556220392, 'c': 10.763290556220392, 'alpha': 109.47122063449069, 'beta': 109.47122063449069, 'gamma': 109.47122063449069, 'volume': 959.8719531108837}, 'properties': {}, 'sites': [{'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.78659388, 0.39175378, 0.13262899], 'xyz': [-6.498293142493029, -3.277792458677283, 1.629429316776457], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.3948401, 0.74087421, 0.60824622], 'xyz': [-3.2777862444885835, -1.6294355309651567, -5.930078043318272], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.65396589, 0.86737101, 0.25912579], 'xyz': [-7.843618016776459, -0.28411687087042936, -2.9363962413227167], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.65396589, 0.25912579, 0.86737101], 'xyz': [-0.28411687087042914, -7.843618016776459, -2.9363962413227167], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.78659388, 0.13262899, 0.39175378], 'xyz': [-3.2777924586772835, -6.498293142493028, 1.6294293167764566], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.3948401, 0.60824622, 0.74087421], 'xyz': [-1.6294355309651563, -3.2777862444885835, -5.930078043318272], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.25912579, 0.65396589, 0.86737101], 'xyz': [-0.28411687087042914, -2.9363962413227167, -7.843618016776459], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.13262899, 0.78659388, 0.39175378], 'xyz': [-3.2777924586772835, 1.6294293167764566, -6.498293142493028], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.60824622, 0.3948401, 0.74087421], 'xyz': [-1.6294355309651563, -5.930078043318272, -3.2777862444885835], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.39175378, 0.78659388, 0.13262899], 'xyz': [-6.498293142493029, 1.629429316776457, -3.277792458677283], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.74087421, 0.3948401, 0.60824622], 'xyz': [-3.2777862444885835, -5.930078043318272, -1.6294355309651567], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.86737101, 0.65396589, 0.25912579], 'xyz': [-7.843618016776459, -2.9363962413227167, -0.28411687087042936], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.74087421, 0.60824622, 0.3948401], 'xyz': [-5.930078043318271, -3.2777862444885835, -1.6294355309651567], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.86737101, 0.25912579, 0.65396589], 'xyz': [-2.9363962413227167, -7.843618016776459, -0.28411687087042914], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.39175378, 0.13262899, 0.78659388], 'xyz': [1.629429316776457, -6.498293142493029, -3.277792458677283], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.60824622, 0.74087421, 0.3948401], 'xyz': [-5.930078043318271, -1.6294355309651567, -3.2777862444885835], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.25912579, 0.86737101, 0.65396589], 'xyz': [-2.9363962413227167, -0.28411687087042914, -7.843618016776459], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.13262899, 0.39175378, 0.78659388], 'xyz': [1.629429316776457, -3.277792458677283, -6.498293142493029], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.21340612, 0.34603411, 0.6051599], 'xyz': [0.28411065668172863, -2.936402455511417, -4.584753169034843], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.6051599, 0.21340612, 0.34603411], 'xyz': [-2.936402455511417, -4.584753169034843, 0.28411065668172863], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.34603411, 0.6051599, 0.21340612], 'xyz': [-4.584753169034843, 0.28411065668172886, -2.936402455511417], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.34603411, 0.21340612, 0.6051599], 'xyz': [0.28411065668172863, -4.584753169034843, -2.936402455511417], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.21340612, 0.6051599, 0.34603411], 'xyz': [-2.936402455511417, 0.28411065668172863, -4.584753169034843], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.6051599, 0.34603411, 0.21340612], 'xyz': [-4.584753169034843, -2.936402455511417, 0.28411065668172886], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [-0.0, -0.0, 0.21736945], 'xyz': [1.350774779915215, -1.350774779915215, -1.350774779915215], 'properties': {'magmom': -0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [-0.0, 0.21736945, -0.0], 'xyz': [-1.350774779915215, 1.350774779915215, -1.350774779915215], 'properties': {'magmom': -0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.78263055, 0.78263055, 0.78263055], 'xyz': [-4.863413920084786, -4.863413920084786, -4.863413920084786], 'properties': {'magmom': -0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.21736945, 0.0, 0.0], 'xyz': [-1.350774779915215, -1.350774779915215, 1.350774779915215], 'properties': {'magmom': -0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.0, 0.0, -0.0], 'xyz': [0.0, 0.0, 0.0], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [-0.0, 0.35932015, 0.22356876], 'xyz': [-0.8435847537472929, 0.8435847537472929, -3.6221816778773173], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.64067985, 0.86424861, 0.64067985], 'xyz': [-5.370603946252707, -2.592007022122684, -5.370603946252707], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.77643124, 0.77643124, 0.13575139], 'xyz': [-8.806195722122682, -0.843584753747293, -0.843584753747293], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.77643124, 0.13575139, 0.77643124], 'xyz': [-0.8435847537472929, -8.806195722122684, -0.8435847537472929], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [-0.0, 0.22356876, 0.35932015], 'xyz': [0.8435847537472929, -0.8435847537472929, -3.6221816778773173], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.64067985, 0.64067985, 0.86424861], 'xyz': [-2.592007022122684, -5.370603946252707, -5.370603946252707], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.13575139, 0.77643124, 0.77643124], 'xyz': [-0.8435847537472929, -0.8435847537472929, -8.806195722122684], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.22356876, -0.0, 0.35932015], 'xyz': [0.8435847537472929, -3.6221816778773173, -0.8435847537472929], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.35932015, -0.0, 0.22356876], 'xyz': [-0.8435847537472929, -3.6221816778773173, 0.8435847537472929], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.86424861, 0.64067985, 0.64067985], 'xyz': [-5.370603946252707, -5.370603946252707, -2.592007022122684], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.35932015, 0.22356876, -0.0], 'xyz': [-3.6221816778773173, -0.8435847537472929, 0.8435847537472929], 'properties': {'magmom': 0.0}, 'label': 'Si'}, {'species': [{'element': 'Si', 'occu': 1}], 'abc': [0.22356876, 0.35932015, -0.0], 'xyz': [-3.6221816778773173, 0.8435847537472929, -0.8435847537472929], 'properties': {'magmom': 0.0}, 'label': 'Si'}]}
)

    expected_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]  # this is not a matrix from get_supercell_size

    try:  # cubic, prefer 90
        new_matrix = get_supercell_size.original(
            structure=structure,
            min_length=18,
            max_length=25,
            prefer_90_degrees=True,
            max_atoms=500,
        )
    except AttributeError:
        warnings.warn(
            message="Falling back to orthorhombic, prefer 90.",
            stacklevel=2,
        )
        try:  # orthorhombic, prefer 90
            new_matrix = get_supercell_size.original(
                structure=structure,
                min_length=18,
                max_length=25,
                prefer_90_degrees=True,
                allow_orthorhombic=True,
                max_atoms=500,
            )
        except AttributeError:
            warnings.warn(
                message="Falling back to orthorhombic.",
                stacklevel=2,
            )
            try:  # orthorhombic
                new_matrix = get_supercell_size.original(
                    structure=structure,
                    min_length=18,
                    max_length=25,
                    prefer_90_degrees=False,
                    allow_orthorhombic=True,
                    max_atoms=500,
                )
            except AttributeError:
                warnings.warn(
                    message="Falling back to a simple supercell size schema. "
                            "Check if this is ok for your use case.",
                    stacklevel=2,
                )
                new_matrix = generate_supercell_matrix(
                    structure, [[3, 0, 0], [0, 3, 0], [0, 0, 3]], max_sites=500
                )

    assert new_matrix == expected_matrix



def test_update_phonon_displacement_maker(memory_jobstore, clean_dir):
    from atomate2.vasp.sets.core import StaticSetGenerator
    from autoplex.data.phonons.flows import TightDFTStaticMakerBigSupercells
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

    lattice_avg = sum(structure.lattice.abc) / 3

    update = update_phonon_displacement_maker(lattice_avg, TightDFTStaticMakerBigSupercells())

    expected = TightDFTStaticMakerBigSupercells(
        name='dft phonon static big supercell',
        input_set_generator=StaticSetGenerator(
            user_incar_settings={
                'IBRION': -1,
                'ISPIN': 1,
                'ISMEAR': 0,
                'ISIF': 3,
                'ENCUT': 700,
                'EDIFF': 1e-07,
                'LAECHG': False,
                'LREAL': False,
                'ALGO': 'Normal',
                'NSW': 0,
                'LCHARG': False,
                'SIGMA': 0.05,
                'ISYM': 0,
                'SYMPREC': 1e-09},
            user_kpoints_settings={'reciprocal_density': 155},  # this is the update we are checking for
            user_potcar_settings={},
            user_potcar_functional=None,
            auto_ismear=True,
            auto_ispin=False,
            auto_lreal=False,
            auto_kspacing=False,
            auto_metal_kpoints=True,
            constrain_total_magmom=False,
            validate_magmom=True,
            use_structure_charge=False,
            sort_structure=True,
            force_gamma=True,
            symprec=0.1,
            vdw=None,
            config_dict={
                'INCAR':
                    {'ALGO': 'Fast',
                     'EDIFF': 1e-05,
                     'EDIFFG': -0.02,
                     'ENAUG': 1360,
                     'ENCUT': 680,
                     'GGA': 'PS',
                     'IBRION': 2,
                     'ISIF': 3,
                     'ISPIN': 2,
                     'ISMEAR': 0,
                     'LORBIT': 11,
                     'LASPH': True,
                     'LDAU': True,
                     'LDAUJ': {'F': {'Co': 0, 'Cr': 0, 'Fe': 0, 'Mn': 0, 'Mo': 0, 'Ni': 0, 'V': 0, 'W': 0},
                               'O': {'Co': 0, 'Cr': 0, 'Fe': 0, 'Mn': 0, 'Mo': 0, 'Ni': 0, 'V': 0, 'W': 0}},
                     'LDAUL': {'F': {'Co': 2, 'Cr': 2, 'Fe': 2, 'Mn': 2, 'Mo': 2, 'Ni': 2, 'V': 2, 'W': 2},
                               'O': {'Co': 2, 'Cr': 2, 'Fe': 2, 'Mn': 2, 'Mo': 2, 'Ni': 2, 'V': 2, 'W': 2}},
                     'LDAUTYPE': 2,
                     'LDAUU': {'F': {'Co': 3.32, 'Cr': 3.7, 'Fe': 5.3, 'Mn': 3.9, 'Mo': 4.38, 'Ni': 6.2, 'V': 3.25, 'W': 6.2},
                               'O': {'Co': 3.32, 'Cr': 3.7, 'Fe': 5.3, 'Mn': 3.9, 'Mo': 4.38, 'Ni': 6.2, 'V': 3.25, 'W': 6.2}},
                     'LDAUPRINT': 1,
                     'LREAL': False,
                     'LMIXTAU': True,
                     'LCHARG': True,
                     'LAECHG': True,
                     'LELF': False,
                     'LWAVE': False,
                     'LVTOT': True,
                     'MAGMOM': {'Ce': 5, 'Ce3+': 1, 'Co': 0.6, 'Co3+': 0.6, 'Co4+': 1, 'Cr': 5, 'Dy3+': 5, 'Er3+': 3, 'Eu': 10, 'Eu2+': 7, 'Eu3+': 6, 'Fe': 5, 'Gd3+': 7, 'Ho3+': 4, 'La3+': 0.6, 'Lu3+': 0.6, 'Mn': 5, 'Mn3+': 4, 'Mn4+': 3, 'Mo': 5, 'Nd3+': 3, 'Ni': 5, 'Pm3+': 4, 'Pr3+': 2, 'Sm3+': 5, 'Tb3+': 6, 'Tm3+': 2, 'V': 5, 'W': 5, 'Yb3+': 1},
                     'NELM': 200,
                     'NSW': 99,
                     'PREC': 'Accurate',
                     'SIGMA': 0.05},
                'KPOINTS': {'reciprocal_density': 64, 'reciprocal_density_metal': 200},
                'POTCAR_FUNCTIONAL': 'PBE_54',
                'POTCAR': {'Ac': 'Ac', 'Ag': 'Ag', 'Al': 'Al', 'Am': 'Am', 'Ar': 'Ar', 'As': 'As', 'At': 'At', 'Au': 'Au', 'B': 'B', 'Ba': 'Ba_sv', 'Be': 'Be', 'Bi': 'Bi_d', 'Br': 'Br', 'C': 'C', 'Ca': 'Ca_sv', 'Cd': 'Cd', 'Ce': 'Ce', 'Cf': 'Cf', 'Cl': 'Cl', 'Cm': 'Cm', 'Co': 'Co', 'Cr': 'Cr_pv', 'Cs': 'Cs_sv', 'Cu': 'Cu', 'Dy': 'Dy_3', 'Er': 'Er_3', 'Eu': 'Eu_2', 'F': 'F', 'Fe': 'Fe', 'Fr': 'Fr_sv', 'Ga': 'Ga_d', 'Gd': 'Gd_3', 'Ge': 'Ge_d', 'H': 'H', 'He': 'He', 'Hf': 'Hf_pv', 'Hg': 'Hg', 'Ho': 'Ho_3', 'I': 'I', 'In': 'In_d', 'Ir': 'Ir', 'K': 'K_sv', 'Kr': 'Kr', 'La': 'La', 'Li': 'Li_sv', 'Lu': 'Lu_3', 'Mg': 'Mg', 'Mn': 'Mn_pv', 'Mo': 'Mo_sv', 'N': 'N', 'Na': 'Na_pv', 'Nb': 'Nb_sv', 'Nd': 'Nd_3', 'Ne': 'Ne', 'Ni': 'Ni', 'Np': 'Np', 'O': 'O', 'Os': 'Os', 'P': 'P', 'Pa': 'Pa', 'Pb': 'Pb_d', 'Pd': 'Pd', 'Pm': 'Pm_3', 'Po': 'Po_d', 'Pr': 'Pr_3', 'Pt': 'Pt', 'Pu': 'Pu', 'Ra': 'Ra_sv', 'Rb': 'Rb_sv', 'Re': 'Re', 'Rh': 'Rh_pv', 'Rn': 'Rn', 'Ru': 'Ru_pv', 'S': 'S', 'Sb': 'Sb', 'Sc': 'Sc_sv', 'Se': 'Se', 'Si': 'Si', 'Sm': 'Sm_3', 'Sn': 'Sn_d', 'Sr': 'Sr_sv', 'Ta': 'Ta_pv', 'Tb': 'Tb_3', 'Tc': 'Tc_pv', 'Te': 'Te', 'Th': 'Th', 'Ti': 'Ti_sv', 'Tl': 'Tl_d', 'Tm': 'Tm_3', 'U': 'U', 'V': 'V_sv', 'W': 'W_sv', 'Xe': 'Xe', 'Y': 'Y_sv', 'Yb': 'Yb_3', 'Zn': 'Zn', 'Zr': 'Zr_sv'}},
            inherit_incar=False,
            lepsilon=False,
            lcalcpol=False),
        write_input_set_kwargs={},
        copy_vasp_kwargs={},
        run_vasp_kwargs={'handlers': ()},
        task_document_kwargs={},
        stop_children_kwargs={},
        write_additional_data={}
    )
    assert update == expected
