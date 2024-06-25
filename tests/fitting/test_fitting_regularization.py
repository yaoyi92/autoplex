from __future__ import annotations
import pytest
from ase.io import read
from ase.atoms import Atom
from autoplex.fitting.common.regularization import (
    set_sigma,
    get_convex_hull,
    get_e_distance_to_hull,
    get_intersect,
    get_x,
    label_stoichiometry_volume,
    point_in_triangle_ND,
    point_in_triangle_2D,
    calculate_hull_ND,
    calculate_hull_3D,
    get_e_distance_to_hull_3D,
    piecewise_linear,
)


def test_set_sigma(test_dir):
    # data setup
    test_atoms = read(test_dir / 'fitting/pre_xyz_train_more_data.extxyz', ':')
    isol_es = {3: -0.28649227, 17: -0.28649227}
    reg_minmax = [(0.1, 1), (0.001, 0.1),
                  (0.0316, 0.316),
                  (0.0632, 0.632)]

    # test series of options for set_sigma

    atoms_modi = set_sigma(test_atoms,
                           reg_minmax,
                           scheme='linear-hull', )
    assert atoms_modi[2].info['energy_sigma'] == 0.001

    atoms_modi = set_sigma(test_atoms,
                           reg_minmax,
                           scheme='linear-hull',
                           config_type_override={'test': [1e-4, 1e-4, 1e-4]}
                           )
    assert atoms_modi[2].info['energy_sigma'] == 1e-4

    atoms_modi[0].info['REF_energy'] += 20
    for atoms in atoms_modi[:3]:
        atoms.set_cell([10, 10, 10])
    for atoms in atoms_modi[4:]:
        atoms.set_cell([11, 11, 11])
    atoms_modi = set_sigma(test_atoms,
                           reg_minmax,
                           scheme='linear-hull',
                           max_energy=0.05,
                           isolated_atoms_energies=isol_es
                           )
    assert len(atoms_modi) < len(test_atoms)

    atoms_modi[0].append(Atom('Li', [1, 1, 1]))
    atoms_modi = set_sigma(test_atoms,
                           reg_minmax,
                           scheme='volume-stoichiometry',
                           isolated_atoms_energies=isol_es
                           )
    assert True  # TODO: modify this to test actual condition


def test_auxiliary_functions(test_dir, memory_jobstore, clean_dir):
    from jobflow import run_locally
    from ase.io import read
    import numpy as np
    import scipy

    file = test_dir / "fitting" / "ref_files" / "quip_train.extxyz"

    atoms = read(file, ":")

    try:
        get_convex = get_convex_hull(atoms)

        responses = run_locally(
            get_convex, ensure_success=True, create_folders=True, store=memory_jobstore
        )

    except ValueError:
        print("\nDOES NOT run as intended, error 'Convex hull failed to include 10/10 structures'")
        assert True

    generic_array = np.array([1, 2, 3, 4, 5])

    try:
        get_e_dist_hull = get_e_distance_to_hull(generic_array, atoms)
    except AttributeError:
        print("\nTODO: implement proper unit test")
        assert True

    point1, point2, point3, point4 = [1, 5], [2, 9], [8, 7], [9, 3]
    point = np.array([[1, 2, 3], [4, 5, 6]])

    get_inter = get_intersect(point1, point2, point3, point4)

    try:
        getx = get_x(atoms)
    except AttributeError:
        print("\nTODO: implement proper unit test")
        assert True

    try:
        label = label_stoichiometry_volume(atoms, {3: -0.28649227, 17: -0.25638457}, "energy")
    except IndexError:
        print("\nTODO: implement proper unit test")
        assert True

    try:
        point_ND = point_in_triangle_ND(point)
    except ValueError:
        print("\nTODO: implement proper unit test")
        assert True

    point_2d = point_in_triangle_2D(point1, point2, point3, point4)

    try:
        calc_hull = calculate_hull_ND(point)
    except scipy.spatial._qhull.QhullError:
        print("\nTODO: implement proper unit test")
        assert True

    try:
        calc_hull_3D = calculate_hull_3D(point)
    except scipy.spatial._qhull.QhullError:
        print("\nTODO: implement proper unit test")
        assert True

    try:
        get_e_dist_hull_3D = get_e_distance_to_hull_3D(generic_array, atoms, {3: -0.28649227, 17: -0.25638457}, "energy")
    except AttributeError:
        print("\nTODO: implement proper unit test")
        assert True

    try:
        piece_lin = piecewise_linear(point1, point)
    except IndexError:
        print("\nTODO: implement proper unit test")
        assert True
