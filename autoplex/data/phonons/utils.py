"""Utility functions for data generation jobs."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from atomate2.common.jobs.phonons import get_supercell_size

if TYPE_CHECKING:
    from atomate2.forcefields.jobs import (
        ForceFieldRelaxMaker,
        ForceFieldStaticMaker,
    )
    from atomate2.vasp.jobs.phonons import PhononDisplacementMaker
    from emmet.core.math import Matrix3D


def ml_phonon_maker_preparation(
    calculator_kwargs: dict,
    relax_maker_kwargs: dict | None,
    static_maker_kwargs: dict | None,
    bulk_relax_maker: ForceFieldRelaxMaker,
    phonon_displacement_maker: ForceFieldStaticMaker,
    static_energy_maker: ForceFieldStaticMaker,
) -> tuple[
    ForceFieldRelaxMaker | None,
    ForceFieldStaticMaker | None,
    ForceFieldStaticMaker | None,
]:
    """
    Prepare the MLPhononMaker for the respective MLIP model.

    bulk_relax_maker: .ForceFieldRelaxMaker or None
        A maker to perform a tight relaxation on the bulk.
        Set to ``None`` to skip the
        bulk relaxation
    static_energy_maker: .ForceFieldStaticMaker or None
        A maker to perform the computation of the DFT energy on the bulk.
        Set to ``None`` to skip the
        static energy computation
    phonon_displacement_maker: .ForceFieldStaticMaker or None
        Maker used to compute the forces for a supercell.
    relax_maker_kwargs: dict
        Keyword arguments that can be passed to the RelaxMaker.
    static_maker_kwargs: dict
        Keyword arguments that can be passed to the StaticMaker.
    """
    if bulk_relax_maker is not None:
        bulk_relax_maker = bulk_relax_maker.update_kwargs(
            update={"calculator_kwargs": calculator_kwargs}
        )
        if relax_maker_kwargs is not None:
            bulk_relax_maker = bulk_relax_maker.update_kwargs(
                update={**relax_maker_kwargs}
            )

    if phonon_displacement_maker is not None:
        phonon_displacement_maker = phonon_displacement_maker.update_kwargs(
            update={"calculator_kwargs": calculator_kwargs}
        )
        if static_maker_kwargs is not None:
            phonon_displacement_maker = phonon_displacement_maker.update_kwargs(
                {**static_maker_kwargs}
            )
    if static_energy_maker is not None:
        static_energy_maker = static_energy_maker.update_kwargs(
            update={"calculator_kwargs": calculator_kwargs}
        )
        if static_maker_kwargs is not None:
            static_energy_maker = static_energy_maker.update_kwargs(
                update={**static_maker_kwargs}
            )

    return bulk_relax_maker, phonon_displacement_maker, static_energy_maker


def update_phonon_displacement_maker(
    lattice, phonon_displacement_maker
) -> PhononDisplacementMaker:
    """
    Update the phonon_displacement_maker.

    Parameters
    ----------
    lattice:
        (Average) lattice of the structure.
    phonon_displacement_maker:
        Maker used to compute the forces for a supercell.

    Returns
    -------
    Updated phonon_displacement_maker

    """
    if lattice > 10:
        density = 350 - 15 * int(round(lattice, 0))
        if lattice > 20:
            density = 50
        phonon_displacement_maker.input_set_generator.user_kpoints_settings = {
            "reciprocal_density": density
        }
    return phonon_displacement_maker


def reduce_supercell_size(
    structure,
    min_length: float = 18,
    max_length: float = 25,
    min_limit: float = 15,
    min_atoms: int = 200,
    max_atoms: int = 500,
) -> Matrix3D:
    """
    Reduce phonopy supercell size.

    Parameters
    ----------
    min_length: float
        min length of the supercell that will be built.
    max_length: float
        max length of the supercell that will be built.
    max_atoms: int
        maximally allowed number of atoms in the supercell.
    min_atoms: int
        minimum number of atoms in the supercell that shall be reached.
    min_limit:
        min_length limit.
    structure: Structure
        pymatgen Structure object.

    Returns
    -------
    supercell_matrix
    """
    best_supercell_matrix = None
    best_num_atoms = max_atoms

    while min_length >= min_limit:
        try:
            warnings.warn(
                message=f"Starting with a cubic supercell with preferred 90°. "
                f"The current min_length is {min_length}.",
                stacklevel=2,
            )
            supercell_matrix = get_supercell_size.original(
                structure=structure,
                min_length=min_length,
                max_length=max_length,
                prefer_90_degrees=True,
                allow_orthorhombic=False,
                max_atoms=max_atoms,
            )
            num_atoms = (structure * supercell_matrix).num_sites
            if num_atoms >= min_atoms:
                return supercell_matrix
            if max_atoms >= num_atoms > best_num_atoms:
                best_supercell_matrix = supercell_matrix
                best_num_atoms = num_atoms
        except AttributeError:
            warnings.warn(
                message=f"Trying cubic supercell. "
                f"The current min_length is {min_length}.",
                stacklevel=2,
            )
        try:
            supercell_matrix = get_supercell_size.original(
                structure=structure,
                min_length=min_length,
                max_length=max_length,
                prefer_90_degrees=False,
                allow_orthorhombic=False,
                max_atoms=max_atoms,
            )
            num_atoms = (structure * supercell_matrix).num_sites
            if num_atoms >= min_atoms:
                return supercell_matrix
            if max_atoms >= num_atoms > best_num_atoms:
                best_supercell_matrix = supercell_matrix
                best_num_atoms = num_atoms
        except AttributeError:
            #     warnings.warn(
            #         message=f"Trying orthorhombic supercell with preferred 90°. "
            #         f"The current min_length is {min_length}.",
            #         stacklevel=2,
            #     )
            # try:
            #     supercell_matrix = get_supercell_size.original(
            #         structure=structure,
            #         min_length=min_length,
            #         max_length=max_length,
            #         prefer_90_degrees=True,
            #         allow_orthorhombic=True,
            #         max_atoms=max_atoms,
            #     )
            #     num_atoms = (structure * supercell_matrix).num_sites
            #     if num_atoms >= min_atoms:
            #         return supercell_matrix
            #     if max_atoms >= num_atoms > best_num_atoms:
            #         best_supercell_matrix = supercell_matrix
            #         best_num_atoms = num_atoms
            # except AttributeError:
            warnings.warn(
                message=f"Trying orthorhombic supercell. "
                f"The current min_length is {min_length}.",
                stacklevel=2,
            )
        try:
            supercell_matrix = get_supercell_size.original(
                structure=structure,
                min_length=min_length,
                max_length=max_length,
                prefer_90_degrees=False,
                allow_orthorhombic=True,
                max_atoms=max_atoms,
            )
            num_atoms = (structure * supercell_matrix).num_sites
            if num_atoms >= min_atoms:
                return supercell_matrix
            if max_atoms >= num_atoms > best_num_atoms:
                best_supercell_matrix = supercell_matrix
                best_num_atoms = num_atoms
        except AttributeError:
            min_length -= 2  # Reduce the min_length by a larger step to reduce run time

    # Return the best supercell found
    if best_supercell_matrix is not None:
        return best_supercell_matrix

    raise ValueError(f"No supercell found with min_length {min_length}.")
