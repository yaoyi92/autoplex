"""fitting using GAP."""

import os

import ase.io
import numpy as np
from ase.neighborlist import NeighborList, natural_cutoffs

from autoplex.fitting.common.utils import energy_remain


def calculate_delta(atoms_db, e_name):
    """
    Calculate delta.

    Parameters
    ----------
    atoms_db
    e_name

    Returns
    -------
    es_var / avg_neigh

    """
    at_ids = [atom.get_atomic_numbers() for atom in atoms_db]
    isol_es = {
        atom.get_atomic_numbers()[0]: atom.info[e_name]
        for atom in atoms_db
        if "config_type" in atom.info and "isol" in atom.info["config_type"]
    }
    es_visol = np.array(
        [
            (atom.info[e_name] - sum([isol_es[j] for j in at_ids[ct]])) / len(atom)
            for ct, atom in enumerate(atoms_db)
        ]
    )
    es_var = np.var(es_visol)
    avg_neigh = np.mean([compute_average_coordination(atom) for atom in atoms_db])
    return es_var / avg_neigh


def compute_average_coordination(atoms):
    """
    Compute average coordination.

    Parameters
    ----------
    atoms

    Returns
    -------
    total_coordination / len(atoms)

    """
    cutoffs = natural_cutoffs(atoms)
    neighbor_list = NeighborList(cutoffs, self_interaction=False, bothways=True)
    neighbor_list.update(atoms)
    total_coordination = sum(
        len(neighbor_list.get_neighbors(index)[0]) for index in range(len(atoms))
    )
    return total_coordination / len(atoms)


def run_command(command):
    """
    Run command.

    Parameters
    ----------
    command

    Returns
    -------
    os.system(command)

    """
    os.system(command)


def gap_fitting(dir, two_body=True, three_body=False, soap=True):
    """
    GAP fitting.

    Parameters
    ----------
    dir
    two_body
    three_body
    soap

    Returns
    -------
    train_error, test_error

    """
    db_atoms = ase.io.read(os.path.join(dir, "train.extxyz"), index=":")
    train_data_path = os.path.join(dir, "train_with_sigma.extxyz")
    test_data_path = os.path.join(dir, "test.extxyz")

    parameters = []

    if two_body:
        delta_2b = calculate_delta(db_atoms, "REF_energy")

        parameters.append(
            f"distance_Nb order=2 compact_clusters=T cutoff=5.0 add_species=T covariance_type=ARD_SE theta_uniform=0.5 "
            f"sparse_method=uniform n_sparse=15 delta={delta_2b} f0=0.0"
        )

        gap_command = (
            "export OMP_NUM_THREADS=32 && gap_fit energy_parameter_name=REF_energy "
            "force_parameter_name=REF_forces virial_parameter_name=REF_virial do_copy_at_file=F at_file={} "
            "gap={{ {} }} default_sigma={{0.0001 0.05 0.05 0}} sparse_jitter=1.0e-10 "
            "gp_file=gap_file.xml".format(train_data_path, " : ".join(parameters))
        )
        run_command(gap_command)

        quip_command = (
            f"export OMP_NUM_THREADS=32 && quip E=T F=T atoms_filename={train_data_path} "
            f"param_filename=gap_file.xml | grep AT | sed 's/AT//' > quip_train.extxyz"
        )
        run_command(quip_command)

    if three_body:
        delta_3b = energy_remain("quip_train.extxyz")
        parameters.append(
            f"distance_Nb order=3 compact_clusters=T cutoff=3.25 covariance_type=ard_se theta_uniform=1.0 "
            f"sparse_method=uniform n_sparse=100 f0=0.0 add_species=T delta={delta_3b}"
        )

        gap_command = (
            "export OMP_NUM_THREADS=32 && gap_fit energy_parameter_name=REF_energy "
            "force_parameter_name=REF_forces virial_parameter_name=REF_virial do_copy_at_file=F "
            "at_file={} gap={{ {} }} default_sigma={{0.0001 0.05 0.05 0}} sparse_jitter=1.0e-10 "
            "gp_file=gap_file.xml".format(train_data_path, " : ".join(parameters))
        )
        run_command(gap_command)

        quip_command = (
            f"export OMP_NUM_THREADS=32 && quip E=T F=T atoms_filename={train_data_path} "
            f"param_filename=gap_file.xml | grep AT | sed 's/AT//' > quip_train.extxyz"
        )
        run_command(quip_command)

    if soap:
        delta_soap = energy_remain("quip_train.extxyz") if two_body or three_body else 1
        parameters.append(
            f"soap l_max=6 n_max=10 atom_sigma=0.5 zeta=4 cutoff=5.0 cutoff_transition_width=0.5 central_weight=1.0 "
            f"n_sparse=1000 f0=0.0 R_mix=T Z_mix=T K=5 covariance_type=dot_product sparse_method=cur_points "
            f"add_species=T delta={delta_soap}"
        )

        gap_command = (
            "export OMP_NUM_THREADS=32 && gap_fit energy_parameter_name=REF_energy "
            "force_parameter_name=REF_forces virial_parameter_name=REF_virial do_copy_at_file=F at_file={} "
            "gap={{ {} }} default_sigma={{0.0001 0.05 0.05 0}} sparse_jitter=1.0e-10 "
            "gp_file=gap_file.xml".format(train_data_path, " : ".join(parameters))
        )
        run_command(gap_command)

        quip_command = (
            f"export OMP_NUM_THREADS=32 && quip E=T F=T atoms_filename={train_data_path} "
            f"param_filename=gap_file.xml | grep AT | sed 's/AT//' > quip_train.extxyz"
        )
        run_command(quip_command)

    # Calculate training error
    train_error = energy_remain("quip_train.extxyz")
    print("Training error of MLIP (eV/at.):", train_error)

    # Calculate testing error
    quip_command = (
        f"export OMP_NUM_THREADS=32 && quip E=T F=T atoms_filename={test_data_path} "
        f"param_filename=gap_file.xml | grep AT | sed 's/AT//' > quip_test.extxyz"
    )
    run_command(quip_command)
    test_error = energy_remain("quip_test.extxyz")
    print("Testing error of MLIP (eV/at.):", test_error)

    return train_error, test_error
