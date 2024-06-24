"""Functions for automatic regularization and weighting of training data."""

# adapted from MorrowChem's RSS routines.
from __future__ import annotations

import traceback
from contextlib import suppress

import numpy as np
from scipy.spatial import ConvexHull, Delaunay


def set_sigma(
    atoms,
    reg_minmax,
    isolated_atoms_energies: dict | None = None,
    scheme="linear-hull",
    energy_name="REF_energy",
    force_name="REF_forces",
    virial_name="REF_virial",
    element_order=None,
    max_energy=20.0,
    config_type_override=None,
):
    """
    Handle automatic regularisation based on distance to convex hull, amongst other things.

    TODO: Need to make sure this works for full multi-stoichiometry systems.

    Parameters
    ----------
    atoms: (list of ase.Atoms)
        list of atoms objects to set reg. for. Usually fitting database
    reg_minmax: (list of tuples)
        list of tuples of (min, max) values for energy, force, virial sigmas
    scheme: (str)
        method to use for regularization. Options are: linear_hull, volume-stoichiometry
        linear_hull: for single-composition system, use 2D convex hull (E, V)
        volume-stoichiometry: for multi-composition system, use 3D convex hull of (E, V, mole-fraction)
    energy_name: (str)
        name of energy key in atoms.info
    force_name: (str)
        name of force key in atoms.arrays
    virial_name: (str)
        name of virial key in atoms.info
    isolated_atoms_energies: (dict)
        dictionary of isolated energies for each atomic number.
        Only needed for volume-x scheme e.g. {14: '-163.0', 8:'-75.0'}
        for SiO2
    element_order: (list)
        list of atomic numbers in order of choice (e.g. [42, 16] for MoS2)
    max_energy: (float)
        ignore any structures with energy above hull greater than this value (eV)
    config_type_override: (dict)
        give custom regularization for specific config types

    e.g. reg_minmax = [(0.1, 1), (0.001, 0.1), (0.0316, 0.316), (0.0632, 0.632)]
    [(emin, emax), (semin, semax), (sfmin, sfmax), (ssvmin, svmax)]

    """
    sigs = [[], [], []]  # type: ignore
    atoms_modi = []

    if config_type_override is None:
        config_type_override = {
            "IsolatedAtom": (1e-4, 0.0, 0.0),
            "dimer": (0.1, 0.5, 0.0),
        }

    for at in atoms:
        for config_type, sigs_override in config_type_override.items():
            if at.info["config_type"] == config_type:
                at.info["energy_sigma"] = sigs_override[0]
                at.info["force_sigma"] = sigs_override[1]
                at.info["virial_sigma"] = sigs_override[2]
                atoms_modi.append(at)

        if at.info["config_type"] == "IsolatedAtom":
            at.calc = None  # TODO side-effect alert
            del at.info["force_sigma"]
            del at.info["virial_sigma"]
            continue

        if at.info["config_type"] == "dimer":
            with suppress(Exception):
                del at.info[virial_name]

            continue

    if scheme == "linear-hull":
        print("Regularising with linear hull")
        hull, p = get_convex_hull(atoms, energy_name=energy_name)
        get_e_distance_func = get_e_distance_to_hull

    elif scheme == "volume-stoichiometry":
        print("Regularising with 3D volume-mole fraction hull")
        if isolated_atoms_energies == {}:
            raise ValueError("Need to supply dictionary of isolated energies.")

        p = label_stoichiometry_volume(
            atoms, isolated_atoms_energies, energy_name, element_order=element_order
        )  # label atoms with volume and mole fraction
        hull = calculate_hull_3D(p)  # calculate 3D convex hull
        get_e_distance_func = get_e_distance_to_hull_3D  # type: ignore

    p = {}
    for group in sorted(
        {  # check if set comprehension is running as supposed to
            at.info.get("gap_rss_group")
            for at in atoms
            if not at.info.get("gap_rss_nonperiodic")
        }
    ):
        p[group] = []

    for at in atoms:
        try:
            # skip non-periodic configs, volume is meaningless
            if "gap_rss_nonperiodic" in at.info and at.info["gap_rss_nonperiodic"]:
                continue
            p[at.info.get("gap_rss_group")].append(at)
        except Exception:
            pass

    for group, atoms_group in p.items():
        print("group:", group)

        for val in atoms_group:
            de = get_e_distance_func(
                hull,
                val,
                energy_name=energy_name,
                isolated_atoms_energies=isolated_atoms_energies,
            )

            if de > max_energy:
                # don't even fit if too high
                continue
            if val.info["config_type"] not in config_type_override:
                if group == "initial":
                    sigs[0].append(reg_minmax[1][1])
                    sigs[1].append(reg_minmax[2][1])
                    sigs[2].append(reg_minmax[3][1])
                    val.info["energy_sigma"] = reg_minmax[1][1]
                    val.info["force_sigma"] = reg_minmax[2][1]
                    val.info["virial_sigma"] = reg_minmax[3][1]
                    atoms_modi.append(val)
                    continue

                if de <= reg_minmax[0][0]:
                    sigs[0].append(reg_minmax[1][0])
                    sigs[1].append(reg_minmax[2][0])
                    sigs[2].append(reg_minmax[3][0])
                    val.info["energy_sigma"] = reg_minmax[1][0]
                    val.info["force_sigma"] = reg_minmax[2][0]
                    val.info["virial_sigma"] = reg_minmax[3][0]
                    atoms_modi.append(val)

                elif de >= reg_minmax[0][1]:
                    sigs[0].append(reg_minmax[1][1])
                    sigs[1].append(reg_minmax[2][1])
                    sigs[2].append(reg_minmax[3][1])
                    val.info["energy_sigma"] = reg_minmax[1][1]
                    val.info["force_sigma"] = reg_minmax[2][1]
                    val.info["virial_sigma"] = reg_minmax[3][1]
                    atoms_modi.append(val)

                else:
                    [e, f, v] = piecewise_linear(
                        de,
                        [
                            (
                                0.1,
                                [reg_minmax[1][0], reg_minmax[2][0], reg_minmax[3][0]],
                            ),
                            (
                                1.0,
                                [reg_minmax[1][1], reg_minmax[2][1], reg_minmax[3][1]],
                            ),
                        ],
                    )
                    sigs[0].append(e)
                    sigs[1].append(f)
                    sigs[2].append(v)
                    val.info["energy_sigma"] = e
                    val.info["force_sigma"] = f
                    val.info["virial_sigma"] = v
                    atoms_modi.append(val)

    labels = ["E", "F", "V"]
    data_type = [np.array(sig) for sig in sigs]  # [e, f, v]

    for label, data in zip(labels, data_type):
        if len(data) == 0:
            print("No automatic regularisation performed (no structures requested)")
            continue
        if label == "E":
            # Report of the regularisation statistics
            print(f"Automatic regularisation statistics for {len(data)} structures:\n")
            print(
                "{:>20s}{:>20s}{:>20s}{:>20s}{:>20s}".format(
                    "", "Mean", "Std", "Nmin", "Nmax"
                )
            )
        print(
            "{:>20s}{:>20.4f}{:>20.4f}{:>20d}{:>20d}".format(
                label,
                data.mean(),
                data.std(),
                (data == data.min()).sum(),
                (data == data.max()).sum(),
            )
        )

    return atoms_modi


def get_convex_hull(atoms, energy_name="energy", **kwargs):
    """
    Calculate simple linear (E,V) convex hull.

    Parameters
    ----------
    atoms: (list)
        list of atoms objects
    energy_name: (str)
        name of energy key in atoms.info (typically a DFT energy)

    Returns
    -------
        the list of points in the convex hull (lower half only),
        and additionally all the points for testing purposes

    """
    p = []
    ct = 0
    for at in atoms:
        if (at.info["config_type"] == "IsolatedAtom") or (
            at.info["config_type"] == "dimer"
        ):
            continue
        try:
            v = at.get_volume() / len(at)
            e = at.info[energy_name] / len(at)
            p.append((v, e))
        except Exception:
            ct += 1
    if ct > 0:
        print(f"Convex hull failed to include {ct}/{len(atoms)} structures")
    p = np.array(p)
    p = p.T[:, np.argsort(p.T[0])].T  # sort in volume axis

    hull = ConvexHull(p)  # generates full convex hull, we only want bottom half
    hull_points = p[hull.vertices]

    min_x_index = np.argmin(hull_points[:, 0])
    max_x_index = np.argmax(hull_points[:, 0])

    lower_half_hull = []
    i = min_x_index

    while True:
        lower_half_hull.append(hull.vertices[i])
        i = (i + 1) % len(hull.vertices)
        if i == max_x_index:
            lower_half_hull.append(hull.vertices[i])
            break

    lower_half_hull_points = p[lower_half_hull]

    lower_half_hull_points = lower_half_hull_points[
        lower_half_hull_points[:, 1] <= np.max(lower_half_hull_points[:, 1])
    ]

    return lower_half_hull_points, p


def get_e_distance_to_hull(hull, at, energy_name="energy", **kwargs):
    """
    Calculate the distance of a structure to the linear convex hull in energy.

    Parameters
    ----------
    hull: (np.array)
        points in the convex hull
    at: (ase.Atoms)
        structure to calculate distance to hull
    energy_name: (str)
        name of energy key in atoms.info (typically a DFT energy)

    """
    volume = at.get_volume() / len(at)
    energy = at.info[energy_name] / len(at)
    tp = np.array([volume, energy])
    hull_ps = hull.points if isinstance(hull, ConvexHull) else hull

    if any(
        np.isclose(hull_ps[:], tp).all(1)
    ):  # if the point is on the hull, return 0.0
        return 0.0

    nearest = np.searchsorted(hull_ps.T[0], tp, side="right")[
        0
    ]  # find the nearest convex hull point

    return (
        energy
        - get_intersect(
            tp,  # get intersection of the vertical line (energy axis)
            tp + np.array([0, 1]),  # and the line between the nearest hull points
            hull_ps[(nearest - 1) % len(hull_ps.T[0])],
            hull_ps[nearest % len(hull_ps.T[0])],
        )[1]
    )


def get_intersect(a1, a2, b1, b2):
    """
    Return the point of intersection of the lines passing through a2,a1 and b2,b1.

    a1: [x, y]
        a point on the first line
    a2: [x, y]
        another point on the first line
    b1: [x, y]
        a point on the second line
    b2: [x, y]
        another point on the second line

    """
    s = np.vstack([a1, a2, b1, b2])  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    if z == 0:  # lines are parallel
        return (float("inf"), float("inf"))
    return x / z, y / z


def get_x(at, element_order=None):
    """
    Calculate the mole-fraction of a structure.

    Parameters
    ----------
    at: (ase.Atoms)
        structure to calculate mole-fraction of
    element_order: (list)
        list of atomic numbers in order of choice (e.g. [42, 16] for MoS2)

    Returns
    -------
    (x2, x3...): (array of float)
        reduced mole-fraction of structure - first element n = 1-sum(others)

    """
    el, cts = np.unique(at.get_atomic_numbers(), return_counts=True)

    if element_order is None and len(el) < 3:  # compatibility with old version
        x = cts[1] / sum(cts) if len(el) == 2 else 1

    else:  # new version, requires element_order, recommended for all new calculations
        if element_order is None:
            element_order = el  # use default order
        not_in = [i for i in element_order if i not in el]
        for i in not_in:
            el = np.insert(el, -1, i)
            cts = np.insert(cts, -1, 0)

        cts = np.array([cts[np.argwhere(el == i).squeeze()] for i in element_order])
        el = np.array([el[np.argwhere(el == i).squeeze()] for i in element_order])

        x = cts[1:] / sum(cts)

    return x


def label_stoichiometry_volume(
    ats, isolated_atoms_energies, e_name, element_order=None
):
    """
    Calculate the stoichiometry, energy, and volume coordinates for forming the convex hull.

    Parameters
    ----------
    ats: (list)
        list of atoms objects
    isolated_atoms_energies: (dict)
        dictionary of isolated atom energies {atomic_number: energy}
    e_name: (str)
        name of energy key in atoms.info (typically a DFT energy)
    element_order: (list)
        list of atomic numbers in order of choice (e.g. [42, 16] for MoS2)

    """
    p = []
    for at in ats:
        try:
            v = at.get_volume() / len(at)
            # make energy relative to isolated atoms
            e = (
                at.info[e_name]
                - sum([isolated_atoms_energies[j] for j in at.get_atomic_numbers()])
            ) / len(at)
            x = get_x(at, element_order=element_order)
            p.append(np.hstack((x, v, e)))
        except Exception:
            traceback.print_exc()
    p = np.array(p)
    return p.T[:, np.argsort(p.T[0])].T


def point_in_triangle_2D(p1, p2, p3, pn):
    """
    Check if a point is inside a triangle in 2D.

    Parameters
    ----------
    p1: (tuple)
        coordinates of first point
    p2: (tuple)
        coordinates of second point
    p3: (tuple)
        coordinates of third point
    pn: (tuple)
        coordinates of point to check

    """
    ep = 1e-4
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x, y = pn

    denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator
    b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator
    c = 1 - a - b

    return (
        0 - ep <= a
        and a <= 1 + ep
        and 0 - ep <= b
        and b <= 1 + ep
        and 0 - ep <= c
        and c <= 1 + ep
    )


def point_in_triangle_ND(pn, *preg):
    """
    Check if a point is inside a region of hyperplanes in N dimensions.

    Make a little convex hull in N-1 D and check this.

    Parameters
    ----------
    pn:
        point to check (in ND)
    *preg:
        list of points defining (in ND) to check against

    """
    hull = Delaunay(preg)
    return hull.find_simplex(pn) >= 0


def calculate_hull_3D(p):
    """
    Calculate the convex hull in 3D.

    Parameters
    ----------
    p:
        point

    Returns
    -------
    convex hull in 3D.

    """
    p0 = np.array(
        [(p[:, i].max() - p[:, i].min()) / 2 + p[:, i].min() for i in range(2)] + [-1e6]
    )  # test point to get the visible facets from below
    pn = np.vstack((p0, p))

    hull = ConvexHull(pn, qhull_options="QG0")
    hull.remove_dim = []

    return hull


def calculate_hull_ND(p):
    """
    Calculate the convex hull in ND (N>=3).

    Parameters
    ----------
    p:
        point

    Returns
    -------
    convex hull in ND.

    """
    p0 = np.array(
        [
            (p[:, i].max() - p[:, i].min()) / 2 + p[:, i].min()
            for i in range(p.shape[1] - 1)
        ]
        + [-1e6]
    )  # test point to get the visible facets from below
    pn = np.vstack((p0, p))
    remove_dim = []

    for i in range(p.shape[1]):
        if np.all(p.T[i, 0] == p.T[i, :]):
            pn = np.delete(pn, i, axis=1)
            print(f"Convex hull lower dimensional - removing dimension {i}")
            remove_dim.append(i)

    hull = ConvexHull(pn, qhull_options="QG0")
    print("done calculating hull")
    hull.remove_dim = remove_dim

    return hull


def get_e_distance_to_hull_3D(
    hull, at, isolated_atoms_energies=None, energy_name="energy", element_order=None
):
    """
    Calculate the energy distance to the convex hull in 3D.

    Parameters
    ----------
    hull:
        convex hull.
    at: (ase.Atoms)
        structure to calculate mole-fraction of
    isolated_atoms_energies: (dict)
        dictionary of isolated atom energies
    energy_name: (str)
        name of energy key in atoms.info (typically a DFT energy)
    element_order: (list)
        list of atomic numbers in order of choice (e.g. [42, 16] for MoS2)

    """
    x = get_x(at, element_order=element_order)
    e = (
        at.info[energy_name]
        - sum([isolated_atoms_energies[j] for j in at.get_atomic_numbers()])
    ) / len(at)
    v = at.get_volume() / len(at)

    sp = np.hstack([x, v, e])
    for i in hull.remove_dim:
        sp = np.delete(sp, i)

    if len(sp[:-1]) == 1:
        # print('doing convexhull analysis in 1D')
        return get_e_distance_to_hull(hull, at, energy_name=energy_name)

    for _ct, visible_facet in enumerate(hull.simplices[hull.good]):
        if point_in_triangle_ND(sp[:-1], *hull.points[visible_facet][:, :-1]):
            n_3 = hull.points[visible_facet]
            e = sp[-1]

            norm = np.cross(n_3[2] - n_3[0], n_3[1] - n_3[0])
            norm = norm / np.linalg.norm(norm)  # plane normal
            D = np.dot(norm, n_3[0])  # plane constant

            return e - (D - norm[0] * sp[0] - norm[1] * sp[1]) / norm[2]

    print("Failed to find distance to hull")
    return 1e6


def piecewise_linear(x, vals):
    """
    Piecewise linear.

    Parameters
    ----------
    x:
        x value.
    vals:
        values

    """
    i = np.searchsorted([v[0] for v in vals], x)
    f0 = (vals[i][0] - x) / (vals[i][0] - vals[i - 1][0])
    return f0 * np.array(vals[i - 1][1]) + (1.0 - f0) * np.array(vals[i][1])
