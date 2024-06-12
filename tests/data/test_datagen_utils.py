from __future__ import annotations

import os
import matplotlib.pyplot as plt
from autoplex.data.common.utils import energy_plot, force_plot, plot_energy_forces, filter_outlier_energy, filter_outlier_forces

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1

fig, ax_list = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(15, 20)

def test_energy_forces(clean_dir, test_dir):
    parent_dir = os.getcwd()
    os.chdir(test_dir / "data")
    energy_forces = plot_energy_forces("regularization 0.1", 0.0005, 0.15)

    assert os.path.isfile("energy_forces.pdf")
    assert os.path.isfile("energy_forces.png")

    for file_name in os.listdir(os.getcwd()):
        if file_name not in ['train.extxyz', 'test.extxyz', 'quip_train.extxyz', 'quip_test.extxyz']:
            os.remove(os.path.join(os.getcwd(), file_name))

    os.chdir(parent_dir)

def test_energy_plot(test_dir, clean_dir):
    parent_dir = os.getcwd()
    os.chdir(test_dir / "data")
    energy = energy_plot("train.extxyz", "quip_train.extxyz", ax_list, "Energy on training data")
    plt.savefig("test_energy.png")
    assert os.path.isfile("test_energy.png")

def test_force_plot(test_dir, clean_dir):
    parent_dir = os.getcwd()
    os.chdir(test_dir / "data")
    force = force_plot("train.extxyz", "quip_train.extxyz", ax_list, "Si", "Force on training data - Si",)
    plt.savefig("test_force.png")
    assert os.path.isfile("test_force.png")

    for file_name in os.listdir(os.getcwd()):
        if file_name not in ['train.extxyz', 'test.extxyz', 'quip_train.extxyz', 'quip_test.extxyz']:
            os.remove(os.path.join(os.getcwd(), file_name))

    os.chdir(parent_dir)

def test_filter_outliers(test_dir, clean_dir):
    parent_dir = os.getcwd()
    os.chdir(test_dir / "data")
    filter_energy = filter_outlier_energy("train.extxyz", "quip_train.extxyz", 0.005)
    filter_forces = filter_outlier_forces("train.extxyz", "quip_train.extxyz", "Si", 0.1)

    assert os.path.isfile("filtered_in_energy.extxyz")
    assert os.path.isfile("filtered_out_energy.extxyz")
    assert os.path.isfile("outliers_energy.extxyz")
    assert os.path.isfile("filtered_in_force.extxyz")
    assert os.path.isfile("filtered_out_force.extxyz")
    assert os.path.isfile("outliers_force.extxyz")

    for file_name in os.listdir(os.getcwd()):
        if file_name not in ['train.extxyz', 'test.extxyz', 'quip_train.extxyz', 'quip_test.extxyz']:
            os.remove(os.path.join(os.getcwd(), file_name))

    os.chdir(parent_dir)