"""Flows to create and check training data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emmet.core.math import Matrix3D
    from pymatgen.core import Structure


from atomate2.forcefields.jobs import (
    ForceFieldRelaxMaker,
    ForceFieldStaticMaker,
    GAPRelaxMaker,
)
from jobflow import Flow, Maker, Response, job

from autoplex.data.common.jobs import (
    convert_to_extxyz,
    generate_randomized_structures,
    get_supercell_job,
    plot_force_distribution,
)

__all__ = ["GenerateTrainingDataForTesting"]


@dataclass
class GenerateTrainingDataForTesting(Maker):
    """Maker for generating training data to test it and check the forces.

    This Maker will first generate training data based on the chosen ML model (default is GAP)
    by randomizing (ase rattle) atomic displacements in supercells of the provided input structures.
    Then it will proceed with MLIP-based Phonon calculations (based on atomate2 PhononMaker), collect
    all structure data in extended xyz files and plot the forces in histograms (per rescaling cell_factor
    and total).

    Parameters
    ----------
    name: str
        Name of the flow.
    bulk_relax_maker: ForceFieldRelaxMaker | None
        Maker for the relax jobs.
    static_energy_maker: ForceFieldStaticMaker | ForceFieldRelaxMaker | None
        Maker for the static jobs.

    """

    name: str = "generate_training_data_for_testing"
    bulk_relax_maker: ForceFieldRelaxMaker | None = None
    static_energy_maker: ForceFieldStaticMaker | ForceFieldRelaxMaker | None = None

    def make(
        self,
        train_structure_list: list[Structure],
        cell_factor_sequence: list[float] | None = None,
        potential_filename: str = "gap.xml",
        n_structures: int = 50,
        rattle_std: float = 0.01,
        relax_cell: bool = True,
        steps: int = 1000,
        supercell_matrix: Matrix3D | None = None,
        config_type: str = "train",
        x_min: int = 0,
        x_max: int = 5,
        bin_width: float = 0.125,
        **relax_kwargs,
    ):
        """
        Generate ase.rattled structures from the training data and returns histogram plots of the forces.

        Parameters
        ----------
        train_structure_list: list[Structure].
            List of pymatgen structures object.
        cell_factor_sequence: list[float]
            list of factor to resize cell parameters.
        potential_filename: str
            param_file_name for :obj:`quippy.potential.Potential()'`.
        n_structures : int.
            Total number of randomly displaced structures to be generated.
        rattle_std: float.
            Rattle amplitude (standard deviation in normal distribution).
            Default=0.01.
        relax_cell : bool
            Whether to allow the cell shape/volume to change during relaxation.
        steps : int
            Maximum number of ionic steps allowed during relaxation.
        supercell_matrix: Matrix3D | None
            The matrix to generate the supercell.
        config_type: str
            configuration type of the data.
        x_min: int
            minimum value for the plot x-axis.
        x_max: int
            maximum value for the plot x-axis.
        bin_width: float
            width of the plot bins.
        relax_kwargs : dict
            Keyword arguments that will get passed to :obj:`Relaxer.relax`.

        Returns
        -------
        matplotlib plots "count vs. forces".
        """
        jobs = []
        if cell_factor_sequence is None:
            cell_factor_sequence = [0.975, 1.0, 1.025, 1.05]
        for structure in train_structure_list:
            if self.bulk_relax_maker is None:
                self.bulk_relax_maker = GAPRelaxMaker(
                    calculator_kwargs={
                        "args_str": "IP GAP",
                        "param_filename": str(potential_filename),
                    },
                    relax_cell=relax_cell,
                    steps=steps,
                )
            if supercell_matrix is None:
                supercell_matrix = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]

            bulk_relax = self.bulk_relax_maker.make(structure=structure)
            jobs.append(bulk_relax)
            supercell = get_supercell_job(
                structure=bulk_relax.output.structure,
                supercell_matrix=supercell_matrix,
            )
            jobs.append(supercell)

            for cell_factor in cell_factor_sequence:
                rand_struc_job = generate_randomized_structures(
                    structure=supercell.output,
                    n_structures=n_structures,
                    volume_custom_scale_factors=[cell_factor],
                    rattle_std=rattle_std,
                )
                jobs.append(rand_struc_job)
                static_conv_jobs = self.static_run_and_convert(
                    rand_struc_job.output,
                    cell_factor,
                    config_type,
                    potential_filename,
                    **relax_kwargs,
                )
                jobs.append(static_conv_jobs)
                plots = plot_force_distribution(
                    cell_factor, static_conv_jobs.output, x_min, x_max, bin_width
                )
                jobs.append(plots)

        return Flow(jobs)  # , plots.output)

    @job
    def static_run_and_convert(
        self,
        structure_list: list[Structure],
        cell_factor: float,
        config_type,
        potential_filename,
        **relax_kwargs,
    ):
        """
        Job for the static runs and the data conversion to the extxyz format.

        Parameters
        ----------
        structure_list: list[Structure].
            List of pymatgen structures object.
        cell_factor: float
            factor to resize cell parameters.
        config_type: str
            configuration type of the data.
        potential_filename: str
            param_file_name for :obj:`quippy.potential.Potential()'`.
        relax_kwargs : dict
            Keyword arguments that will get passed to :obj:`Relaxer.relax`.

        """
        jobs = []
        for rand_struc in structure_list:
            if relax_kwargs == {}:
                relax_kwargs = {
                    "interval": 50000,
                    "fmax": 0.5,
                    "traj_file": rand_struc.reduced_formula
                    + "_"
                    + f"{cell_factor}".replace(".", "")
                    + ".pkl",
                }
            if self.static_energy_maker is None:
                self.static_energy_maker = GAPRelaxMaker(
                    calculator_kwargs={
                        "args_str": "IP GAP",
                        "param_filename": str(potential_filename),
                    },
                    relax_cell=False,
                    relax_kwargs=relax_kwargs,
                    steps=1,
                )
            static_run = self.static_energy_maker.make(structure=rand_struc)
            jobs.append(static_run)
            conv_job = convert_to_extxyz(
                static_run.output,
                rand_struc.reduced_formula
                + "_"
                + f"{cell_factor}".replace(".", "")
                + ".pkl",
                config_type,
                f"{cell_factor}".replace(".", ""),
            )
            jobs.append(conv_job)

        return Response(replace=Flow(jobs), output=conv_job.output)
