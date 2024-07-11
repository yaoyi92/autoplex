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
    structure = Structure(  # mp-1203790
        lattice=[[-5.183318, -8.977762, 0.000000], [-5.183315, 8.977761, -0.000000], [0.000000, 0.000000, -16.970272]],
        species=["Si"]*68,
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

    supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

    expected_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]

    new_matrix = generate_supercell_matrix(structure=structure, supercell_matrix=supercell_matrix)

    assert new_matrix == expected_matrix



