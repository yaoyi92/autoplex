"""Flows to perform automatic data generation, fitting, and benchmarking of ML potentials."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atomate2.common.schemas.phonons import PhononBSDOSDoc
    from atomate2.vasp.jobs.base import BaseVaspMaker
    from emmet.core.math import Matrix3D
    from pymatgen.core.structure import Structure

from jobflow import Flow, Maker

from autoplex.auto.phonons.jobs import (
    MLPhononMaker,
    dft_phonopy_gen_data,
    dft_random_gen_data,
    get_iso_atom,
)
from autoplex.benchmark.phonons.flows import PhononBenchmarkMaker
from autoplex.benchmark.phonons.jobs import write_benchmark_metrics
from autoplex.data.phonons.flows import DFTPhononMaker, TightDFTStaticMaker
from autoplex.fitting.common.flows import MLIPFitMaker

__all__ = ["CompleteDFTvsMLBenchmarkWorkflow"]


# Volker's idea: provide several default flows with different setting/setups


@dataclass
class CompleteDFTvsMLBenchmarkWorkflow(Maker):
    """
    Maker to construct a DFT (VASP) based dataset, composed of the following two configuration types.

    1) single atom displaced supercells (based on the atomate2 PhononMaker subroutines)
    2) supercells with randomly displaced atoms (based on the ase rattled function).
    Machine-learned interatomic potential(s) are then fitted on the dataset, followed by
    benchmarking the resulting potential(s) to DFT (VASP) level using the provided benchmark
    structure(s) and comparing the respective DFT and MLIP-based Phonon calculations.
    The benchmark metrics are provided in form of a phonon band structure comparison and
    q-point-wise phonons RMSE plots, as well as a summary text file.

    Parameters
    ----------
    name : str
        Name of the flow produced by this maker.
    add_dft_phonon_struct: bool.
        If True, will add displaced supercells via phonopy for DFT calculation.
    add_dft_random_struct: bool.
        If True, will add randomly distorted structures for DFT calculation.
    add_rss_struct: bool.
        If True, will add RSS generated structures for DFT calculation.
        n_struct: int.
        The total number of randomly displaced structures to be generated.
    phonon_displacement_maker: BaseVaspMaker
        Maker used to compute the forces for a supercell.
    n_struct: int.
        The total number of randomly displaced structures to be generated.
    displacements: list[float]
        displacement distance for phonons
    min_length: float
        min length of the supercell that will be built
    symprec: float
        Symmetry precision to use in the
        reduction of symmetry to find the primitive/conventional cell
        (use_primitive_standard_structure, use_conventional_standard_structure)
        and to handle all symmetry-related tasks in phonopy
    uc: bool.
        If True, will generate randomly distorted structures (unitcells)
        and add static computation jobs to the flow.
    cell_factor_sequence: list[float]
        list of factors to resize cell parameters. Default is [0.975, 1.0, 1.025, 1.05].
    std_dev: float
        Standard deviation std_dev for normal distribution to draw numbers from
        to generate the rattled structures.
    supercell_matrix: Matrix3D or None
        The matrix to construct the supercell.
    ml_models: list[str]
        list of the ML models to be used. Default is GAP.
    """

    name: str = "add_data"
    add_dft_phonon_struct: bool = True
    add_dft_random_struct: bool = True
    add_rss_struct: bool = False

    phonon_displacement_maker: BaseVaspMaker = field(
        default_factory=TightDFTStaticMaker
    )
    n_struct: int = 1
    displacements: list[float] = field(default_factory=lambda: [0.01])
    min_length: int = 20
    symprec: float = 1e-4
    uc: bool = False
    cell_factor_sequence: list[float] | None = None
    std_dev: float = 0.01
    supercell_matrix: Matrix3D | None = None
    ml_models: list[str] = field(default_factory=lambda: ["GAP"])

    def make(
        self,
        structure_list: list[Structure],
        mp_ids,
        split_ratio: float = 0.4,
        f_max: float = 40.0,
        pre_xyz_files: list[str] | None = None,
        pre_database_dir: str | None = None,
        atom_wise_regularization_parameter: float = 0.1,
        f_min: float = 0.01,  # unit: eV Å-1
        atom_wise_regularization: bool = True,
        auto_delta: bool = True,
        dft_references: list[PhononBSDOSDoc] | None = None,
        benchmark_structures: list[Structure] | None = None,
        benchmark_mp_ids: list[str] | None = None,
        **fit_kwargs,
    ):
        """
        Make flow for constructing the dataset, fitting the potentials and performing the benchmarks.

        Parameters
        ----------
        structure_list:
            list of pymatgen structures.
        mp_ids:
            materials project id.
        split_ratio: float.
            Parameter to divide the training set and the test set.
            A value of 0.1 means that the ratio of the training set to the test set is 9:1.
        f_max: float
            Maximally allowed force in the data set.
        pre_xyz_files: list[str] or None
            names of the pre-database train xyz file and test xyz file.
        pre_database_dir:
            the pre-database directory.
        atom_wise_regularization_parameter: float
            regularization value for the atom-wise force components.
        f_min: float
            minimal force cutoff value for atom-wise regularization.
        atom_wise_regularization: bool
            for including atom-wise regularization.
        auto_delta: bool
            automatically determine delta for 2b, 3b and soap terms.
        dft_references: list[PhononBSDOSDoc] | None
            DFT reference file containing the PhononBSDOCDoc object.
        benchmark_structures: list[Structure] | None
            pymatgen structure for benchmarking.
        benchmark_mp_ids: list[str] | None
            Materials Project ID of the benchmarking structure.

        """
        flows = []
        fit_input = {}
        collect = []

        for structure, mp_id in zip(structure_list, mp_ids):
            if self.add_dft_random_struct:
                addDFTrand = self.add_dft_random(
                    structure,
                    mp_id,
                    self.phonon_displacement_maker,
                    self.n_struct,
                    self.uc,
                    self.cell_factor_sequence,
                    self.std_dev,
                    self.supercell_matrix,
                )
                flows.append(addDFTrand)
                fit_input.update({mp_id: addDFTrand.output})
            if self.add_dft_phonon_struct:
                addDFTphon = self.add_dft_phonons(
                    structure,
                    self.displacements,
                    self.symprec,
                    self.phonon_displacement_maker,
                    self.min_length,
                )
                flows.append(addDFTphon)
                fit_input.update({mp_id: addDFTphon.output})
            if self.add_dft_random_struct and self.add_dft_phonon_struct:
                fit_input.update({mp_id: {**addDFTrand.output, **addDFTphon.output}})
            if self.add_rss_struct:
                raise NotImplementedError

        isoatoms = get_iso_atom(structure_list)
        flows.append(isoatoms)

        if pre_xyz_files is None:
            fit_input.update(
                {"isolated_atom": {"iso_atoms_dir": [isoatoms.output["dirs"]]}}
            )

        for ml_model in self.ml_models:
            add_data_fit = MLIPFitMaker(mlip_type=ml_model).make(
                species_list=isoatoms.output["species"],
                isolated_atoms_energy=isoatoms.output["energies"],
                fit_input=fit_input,
                split_ratio=split_ratio,
                f_max=f_max,
                pre_xyz_files=pre_xyz_files,
                pre_database_dir=pre_database_dir,
                atom_wise_regularization_parameter=atom_wise_regularization_parameter,
                f_min=f_min,
                atom_wise_regularization=atom_wise_regularization,
                auto_delta=auto_delta,
                **fit_kwargs,
            )
            flows.append(add_data_fit)

        bm_outputs = []

        if (benchmark_structures is not None) and (benchmark_mp_ids is not None):
            for ibenchmark_structure, benchmark_structure in enumerate(
                benchmark_structures
            ):
                add_data_ml_phonon = MLPhononMaker(
                    min_length=self.min_length,
                ).make_from_ml_model(
                    structure=benchmark_structure,
                    ml_model=add_data_fit.output["mlip_xml"],
                )
                flows.append(add_data_ml_phonon)

                if dft_references is None and benchmark_mp_ids is not None:
                    if (
                        benchmark_mp_ids[ibenchmark_structure] in mp_ids
                    ) and self.add_dft_phonon_struct:
                        dft_references = fit_input[
                            benchmark_mp_ids[ibenchmark_structure]
                        ]["phonon_data"]["001"]
                    elif (
                        benchmark_mp_ids[ibenchmark_structure] not in mp_ids
                    ) or (  # else?
                        self.add_dft_phonon_struct is False
                    ):
                        dft_phonons = DFTPhononMaker(
                            symprec=self.symprec,
                            phonon_displacement_maker=self.phonon_displacement_maker,
                            born_maker=None,
                            min_length=self.min_length,
                        ).make(structure=benchmark_structure)
                        flows.append(dft_phonons)
                        dft_references = dft_phonons.output

                    add_data_bm = PhononBenchmarkMaker(name="Benchmark").make(
                        structure=benchmark_structure,
                        benchmark_mp_id=benchmark_mp_ids[ibenchmark_structure],
                        ml_phonon_task_doc=add_data_ml_phonon.output,
                        dft_phonon_task_doc=dft_references,
                    )
                else:
                    add_data_bm = PhononBenchmarkMaker(name="Benchmark").make(
                        structure=benchmark_structure,
                        benchmark_mp_id=benchmark_mp_ids[ibenchmark_structure],
                        ml_phonon_task_doc=add_data_ml_phonon.output,
                        dft_phonon_task_doc=dft_references[ibenchmark_structure],
                    )
                flows.append(add_data_bm)
                collect.append(add_data_bm.output)

                collect_bm = write_benchmark_metrics(
                    benchmark_structure=benchmark_structure,
                    mp_id=benchmark_mp_ids[ibenchmark_structure],
                    rmse=collect,
                    displacements=self.displacements,
                )
                flows.append(collect_bm)
                bm_outputs.append(collect_bm.output)
        return Flow(flows, bm_outputs)

    def add_dft_phonons(
        self,
        structure: Structure,
        displacements: list[float],
        symprec: float,
        phonon_displacement_maker: BaseVaspMaker,
        min_length: float,
    ):
        """Add DFT phonon runs for reference structures.

        Parameters
        ----------
        structure: Structure
            pymatgen Structure object
        displacements: list[float]
           displacement distance for phonons
        symprec: float
            Symmetry precision to use in the
            reduction of symmetry to find the primitive/conventional cell
            (use_primitive_standard_structure, use_conventional_standard_structure)
            and to handle all symmetry-related tasks in phonopy
        phonon_displacement_maker: BaseVaspMaker
            Maker used to compute the forces for a supercell.
        min_length: float
             min length of the supercell that will be built
        """
        additonal_dft_phonon = dft_phonopy_gen_data(
            structure, displacements, symprec, phonon_displacement_maker, min_length
        )

        return Flow(
            additonal_dft_phonon,  # flows
            output={
                "phonon_dir": additonal_dft_phonon.output["dirs"],
                "phonon_data": additonal_dft_phonon.output["data"],
            },
        )

    def add_dft_random(
        self,
        structure: Structure,
        mp_id: str,
        phonon_displacement_maker: BaseVaspMaker,
        n_struct: int = 1,
        uc: bool = False,
        cell_factor_sequence: list[float] | None = None,
        std_dev: float = 0.01,
        supercell_matrix: Matrix3D | None = None,
    ):
        """Add DFT phonon runs for randomly displaced structures.

        Parameters
        ----------
        structure: Structure
            pymatgen Structure object
        mp_id: str
            materials project id
        phonon_displacement_maker: BaseVaspMaker
            Maker used to compute the forces for a supercell.
        n_struct: int.
            The total number of randomly displaced structures to be generated.
        uc: bool.
            If True, will generate randomly distorted structures (unitcells)
            and add static computation jobs to the flow.
        cell_factor_sequence: list[float]
            list of factors to resize cell parameters.
        std_dev: float
            Standard deviation std_dev for normal distribution to draw numbers from
            to generate the rattled structures.
        supercell_matrix: Matrix3D or None
            The matrix to construct the supercell.
        """
        additonal_dft_random = dft_random_gen_data(
            structure,
            mp_id,
            phonon_displacement_maker,
            n_struct,
            uc,
            cell_factor_sequence,
            std_dev,
            supercell_matrix,
        )
        return Flow(
            additonal_dft_random,
            output={"rand_struc_dir": additonal_dft_random.output},
        )
