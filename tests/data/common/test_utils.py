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
)


fig, ax_list = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(15, 20)


def test_energy_forces(clean_dir, test_dir):
    parent_dir = os.getcwd()
    os.chdir(test_dir / "data" / "ref_data")
    plot_energy_forces(
        title="regularization 0.1",
        energy_limit=0.0005,
        force_limit=0.15,
        train_name='train_Si.extxyz',
        test_name='test_Si.extxyz'
    )
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
    force = force_plot("train_Si.extxyz", "quip_train_Si.extxyz", ax_list, "Si", "Force on training data - Si", )
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


def test_scale_cell(vasp_test_dir):
    from autoplex.data.common.utils import scale_cell

    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    scaled_cell = scale_cell(structure=structure, volume_scale_factor_range=[0.95, 1.05])
    scaled_cell_1 = scale_cell(structure=structure, volume_scale_factor_range=[0.95, 1.0])
    scaled_cell_1_2 = scale_cell(structure=structure, volume_scale_factor_range=[0.95, 1])
    scaled_cell_2 = scale_cell(structure=structure, volume_custom_scale_factors=[0.95, 1.05, 1.10])
    scaled_cell_2_2 = scale_cell(structure=structure, volume_custom_scale_factors=[0.95, 1.0, 1.05, 1.10])

    assert len(scaled_cell) == 11
    assert len(scaled_cell_1) == 10
    assert len(scaled_cell_1_2) == 10
    assert len(scaled_cell_2) == 3
    assert len(scaled_cell_2_2) == 4
