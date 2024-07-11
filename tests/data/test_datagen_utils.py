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
