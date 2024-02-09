"""Flows perform automatic data generation, fitting, and benchmarking of ML potentials."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atomate2.common.schemas.phonons import PhononBSDOSDoc
    from atomate2.vasp.jobs.base import BaseVaspMaker
    from emmet.core.math import Matrix3D
    from pymatgen.core.structure import Structure

from jobflow import Flow, Maker

from autoplex.auto.jobs import (
    dft_phonopy_gen_data,
    dft_random_gen_data,
    get_iso_atom,
    get_phonon_ml_calculation_jobs,
)
from autoplex.benchmark.flows import PhononBenchmarkMaker
from autoplex.benchmark.jobs import write_benchmark_metrics
from autoplex.data.flows import DFTPhononMaker, TightDFTStaticMaker
from autoplex.fitting.flows import MLIPFitMaker

__all__ = [
    "CompleteDFTvsMLBenchmarkWorkflow",
    "DFTDataGenerationFlow",
    "PhononDFTMLBenchmarkFlow",
]


# Volker's idea: provide several default flows with different setting/setups
# TODO TaskDocs


@dataclass
class CompleteDFTvsMLBenchmarkWorkflow(Maker):
    """
    Maker to add more data to existing dataset (.xyz file).

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
    """

    # TODO docstrings
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
    supercell_matrix: Matrix3D | None = None
    uc: bool = False

    def make(
        self,
        structure_list: list[Structure],
        mp_ids,
        xyz_file: str | None = None,
        dft_references: PhononBSDOSDoc | None = None,
        benchmark_structures: list[Structure] | None = None,
        benchmark_mp_ids: list[str] | None = None,
        **fit_kwargs,
    ):
        """
        Make flow for adding data to the dataset.

        Parameters
        ----------
        structure_list:
            list of pymatgen structures.
        mp_ids:
            materials project id.
        xyz_file:
            the already existing training data xyz file.
        dft_references:
            DFT reference file containing the PhononBSDOCDoc object.
        benchmark_structures:
            pymatgen structure for benchmarking.
        benchmark_mp_ids:
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

        if xyz_file is None:
            fit_input.update(
                {"isolated_atom": {"iso_atoms_dir": [isoatoms.output["dirs"]]}}
            )

        add_data_fit = PhononDFTMLFitFlow().make(
            species=isoatoms.output["species"],
            isolated_atoms_energy=isoatoms.output["energies"],
            xyz_file=xyz_file,
            fit_input=fit_input,
            **fit_kwargs,
        )
        flows.append(add_data_fit)

        bm_outputs = []

        if (benchmark_structures is not None) and (benchmark_mp_ids is not None):
            for ibenchmark_structure, benchmark_structure in enumerate(
                benchmark_structures
            ):
                add_data_ml_phonon = get_phonon_ml_calculation_jobs(
                    structure=benchmark_structure,
                    min_length=self.min_length,
                    ml_dir=add_data_fit.output,
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

                    add_data_bm = PhononDFTMLBenchmarkFlow(name="addDataBM").make(
                        structure=benchmark_structure,
                        benchmark_mp_id=benchmark_mp_ids[ibenchmark_structure],
                        ml_phonon_task_doc=add_data_ml_phonon.output,
                        dft_phonon_task_doc=dft_references,
                    )
                else:
                    add_data_bm = PhononDFTMLBenchmarkFlow(name="addDataBM").make(
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
        displacements:
           displacement distance for phonons
        symprec:
            Symmetry precision to use in the
            reduction of symmetry to find the primitive/conventional cell
            (use_primitive_standard_structure, use_conventional_standard_structure)
            and to handle all symmetry-related tasks in phonopy
        phonon_displacement_maker:
            Maker used to compute the forces for a supercell.
        min_length:
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
        n_struct: int,
        uc: bool,
        supercell_matrix: Matrix3D | None = None,
    ):
        """Add DFT phonon runs for randomly displaced structures.

        Parameters
        ----------
        structure: Structure
            pymatgen Structure object
        mp_id:
            materials project id
        n_struct: int.
            The total number of randomly displaced structures to be generated.
        phonon_displacement_maker:
            Maker used to compute the forces for a supercell.
        uc: bool.
            If True, will generate randomly distorted structures (unitcells)
            and add static computation jobs to the flow
        supercell_matrix: Matrix3D or None
            The matrix to construct the supercell.
        """
        additonal_dft_random = dft_random_gen_data(
            structure, mp_id, phonon_displacement_maker, n_struct, uc, supercell_matrix
        )

        return Flow(
            additonal_dft_random,  # flows
            output={"rand_struc_dir": additonal_dft_random.output},
        )


@dataclass
class DFTDataGenerationFlow(Maker):
    """
    Maker to generate DFT reference database to be used for fitting ML potentials.

    The maker will use phonopy to create displacements according to the finite displacement method.
    In addition, randomly distorted structures are added to the dataset.

    Parameters
    ----------
    name : str
        Name of the flow produced by this maker.
    phonon_displacement_maker : .BaseVaspMaker or None
        Maker used to compute the forces for a supercell.
    n_struct: int.
        The total number of randomly displaced structures to be generated.
    displacements: list[float]
        list of phonon displacement
    min_length: float
        min length of the supercell that will be built
    symprec : float
        Symmetry precision to use in the
        reduction of symmetry to find the primitive/conventional cell
        (use_primitive_standard_structure, use_conventional_standard_structure)
        and to handle all symmetry-related tasks in phonopy
    uc: bool.
        If True, will generate randomly distorted supercells structures
        and add static computation jobs to the flow
    """

    name: str = "datagen"
    phonon_displacement_maker: BaseVaspMaker = field(
        default_factory=TightDFTStaticMaker
    )
    n_struct: int = 1
    displacements: list[float] = field(default_factory=lambda: [0.01])
    min_length: int = 20
    symprec: float = 1e-4
    uc: bool = False
    supercell_matrix: Matrix3D | None = None

    def make(self, structure: Structure, mp_id):
        """
        Make flow for data generation.

        Parameters
        ----------
        structure: Structure
            pymatgen Structure object
        mp_id:
            materials project id
        """
        # TODO later adding: for i no. of potentials

        dft_phonon = dft_phonopy_gen_data(
            structure,
            self.displacements,
            self.symprec,
            self.phonon_displacement_maker,
            self.min_length,
        )

        if self.n_struct != 0:
            dft_random = dft_random_gen_data(
                structure,
                mp_id,
                self.phonon_displacement_maker,
                self.n_struct,
                self.uc,
                self.supercell_matrix,
            )

            flows = [dft_phonon, dft_random]
            output = {
                "rand_struc_dir": dft_random.output,
                "phonon_dir": dft_phonon.output["dirs"],
                "phonon_data": dft_phonon.output["data"],
            }

        else:
            flows = [dft_phonon]
            output = {
                "phonon_dir": dft_phonon.output["dirs"],
                "phonon_data": dft_phonon.output["data"],
            }

        return Flow(
            flows,  # flows
            output=output,
        )


@dataclass
class PhononDFTMLFitFlow(Maker):
    """
    Maker to fit several types of ML potentials (GAP, ACE etc.) based on DFT data.

    (Currently only the subroutines for GAP are implemented).

    Parameters
    ----------
    name : str
        Name of the flow produced by this maker.

    """

    name: str = "ML fit"

    def make(
        self,
        species,
        isolated_atoms_energy,
        fit_input: dict,
        xyz_file: str | None = None,
        **fit_kwargs,
    ):
        """
        Make flow for to fit potential.

        Parameters
        ----------
        species: list[Species]
            List of species
        isolated_atoms_energy: list.
            Isolated atoms energy list
        fit_input: list.
            Mixed list of dictionary and lists of the fit input data.
        xyz_file: str or None
            a possibly already existing xyz file
        fit_kwargs : dict.
            dict including gap fit keyword args.
        """
        flows = []

        ml_fit_flow = MLIPFitMaker(name="GAP").make(
            species_list=species,
            iso_atom_energy=isolated_atoms_energy,
            fit_input=fit_input,
            xyz_file=xyz_file,
            **fit_kwargs,
        )
        flows.append(ml_fit_flow)

        return Flow(flows, ml_fit_flow.output)


# We need to extend this flow to run over more than one structure.
# I am not sure why it even is a flow
@dataclass
class PhononDFTMLBenchmarkFlow(Maker):
    """
    Maker for benchmarking ML potential.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    """

    name: str = "ML DFT benchmark"

    def make(
        self,
        structure: Structure,
        benchmark_mp_id,
        ml_phonon_task_doc: PhononBSDOSDoc,
        dft_phonon_task_doc: PhononBSDOSDoc,
    ):
        """
        Create flow to benchmark the ML potential.

        Parameters
        ----------
        structure: Structure
            Structure used for benchmark
        benchmark_mp_id: str.
            Material project id string
        ml_phonon_task_doc: PhononBSDOSDoc
            Phonon task doc from ML potential consisting of pymatgen band-structure object
        dft_phonon_task_doc: PhononBSDOSDoc
            Phonon task doc from DFT runs consisting of pymatgen band-structure object
        """
        flows = []

        benchmark = PhononBenchmarkMaker(name="Benchmark").make(
            structure=structure,
            benchmark_mp_id=benchmark_mp_id,
            ml_phonon_bs=ml_phonon_task_doc.phonon_bandstructure,  # TODO take BS at top lvl?
            dft_phonon_bs=dft_phonon_task_doc.phonon_bandstructure,
        )
        flows.append(benchmark)

        return Flow(flows, benchmark.output)
