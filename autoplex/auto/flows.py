"""Flows perform automatic data generation, fitting, and benchmarking of ML potentials."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atomate2.common.schemas.phonons import PhononBSDOSDoc
    from atomate2.vasp.jobs.base import BaseVaspMaker
    from emmet.core.math import Matrix3D
    from pymatgen.core.structure import Structure

from atomate2.common.jobs.phonons import PhononDisplacementMaker
from atomate2.vasp.flows.phonons import PhononMaker as DFTPhononMaker
from atomate2.vasp.powerups import (
    update_user_incar_settings,
)
from jobflow import Flow, Maker

from autoplex.auto.jobs import (
    dft_phonopy_gen_data,
    dft_random_gen_data,
    get_iso_atom,
    get_phonon_ml_calculation_jobs,
)
from autoplex.benchmark.flows import PhononBenchmarkMaker
from autoplex.benchmark.jobs import write_benchmark_metrics
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
    Maker to calculate harmonic phonons with DFT, fit GAP and benchmark the results.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker
    n_struct: int.
        The total number of randomly displaced structures to be generated.
    displacements: List[float]
        displacement distance for phonons
    symprec : float
        Symmetry precision to use in the
        reduction of symmetry to find the primitive/conventional cell
        (use_primitive_standard_structure, use_conventional_standard_structure)
        and to handle all symmetry-related tasks in phonopy
    sc: bool.
        If True, will generate randomly distorted supercells structures
        and add static computation jobs to the flow

    """

    name: str = "complete_workflow"
    n_struct: int = 1
    displacements: list[float] = field(default_factory=lambda: [0.01])
    min_length: int = 20
    symprec: float = 1e-4
    sc: bool = False
    supercell_matrix: Matrix3D | None = None

    def make(
        self,
        structure_list: list[Structure],
        mp_ids,
        phonon_displacement_maker,
        benchmark_structure: Structure,
        mp_id,
    ):
        """
        Make the complete workflow for DFT vs. ML benchmarking.

        Parameters
        ----------
        structure_list: List[Structure]
            list of pymatgen structures
        mp_ids : list.
            list of materials project ids
        phonon_displacement_maker : .BaseVaspMaker
            Maker used to compute the forces for a supercell.
        benchmark_structure: Structure.
            Structure used for benchmarking.
        mp_id: str
            materials project ID corresponding to the benchmark structure
        """
        flows = []
        datagen = {}
        collect = []
        isoatoms = get_iso_atom(structure_list)
        flows.append(isoatoms)

        for struc_i, structure in enumerate(structure_list):
            mp_id = mp_ids[struc_i]
            autoplex_datagen = DFTDataGenerationFlow(
                name="datagen",
                phonon_displacement_maker=phonon_displacement_maker,
                n_struct=self.n_struct,
                displacements=self.displacements,
                min_length=self.min_length,
                symprec=self.symprec,
                sc=self.sc,
                supercell_matrix=self.supercell_matrix,
            ).make(structure=structure, mp_id=mp_id)
            flows.append(autoplex_datagen)
            datagen.update({mp_id: autoplex_datagen.output})

        autoplex_fit = PhononDFTMLFitFlow().make(
            species=isoatoms.output["species"],
            isolated_atoms_energy=isoatoms.output["energies"],
            fit_input=datagen,
        )
        flows.append(autoplex_fit)

        autoplex_ml_phonon = get_phonon_ml_calculation_jobs(
            structure=benchmark_structure,
            min_length=self.min_length,
            ml_dir=autoplex_fit.output,
        )
        flows.append(autoplex_ml_phonon)
        if mp_id not in mp_ids:
            dft_phonons = DFTPhononMaker(
                symprec=self.symprec,
                phonon_displacement_maker=phonon_displacement_maker,
                born_maker=None,
                min_length=self.min_length,
            ).make(structure=benchmark_structure)
            dft_phonons = update_user_incar_settings(dft_phonons, {"NPAR": 4})
            flows.append(dft_phonons)

            dft_reference = dft_phonons.output
        else:
            dft_reference = datagen[mp_id]["phonon_data"]["001"]

        autoplex_bm = PhononDFTMLBenchmarkFlow(name="testBM").make(
            structure=benchmark_structure,
            mp_id=mp_id,
            ml_phonon_task_doc=autoplex_ml_phonon.output,
            dft_phonon_task_doc=dft_reference,
        )
        flows.append(autoplex_bm)
        collect.append(autoplex_bm.output)

        collect_bm = write_benchmark_metrics(
            benchmark_structure=benchmark_structure,
            mp_id=mp_id,
            rmse=collect,
            displacements=self.displacements,
        )
        flows.append(collect_bm)

        return Flow(flows)


@dataclass
class AddDataToDataset(Maker):
    """
    Maker to add more data to existing daaset (.xyz file).

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

    name: str = "add_data"
    add_dft_phonon_struct: bool = True
    add_dft_random_struct: bool = True
    add_rss_struct: bool = False

    phonon_displacement_maker: BaseVaspMaker = field(
        default_factory=PhononDisplacementMaker
    )
    n_struct: int = 1
    displacements: list[float] = field(default_factory=lambda: [0.01])
    min_length: int = 20
    symprec: float = 1e-4
    sc: bool = False
    supercell_matrix: Matrix3D | None = None

    def make(
        self,
        structure_list: list[Structure],
        mp_ids,
        xyz_file,
        benchmark_structure: Structure,
        mp_id,
    ):
        """
        Make flow for adding data to the dataset.

        Parameters
        ----------
        structure_list: List[Structure]
            list of pymatgen structures
        mp_ids:
            materials project id
        xyz_file:
            the already existing training data xyz file
        benchmark_structure: Structure
            pymatgen structure for benchmarking.
        mp_id:
            Materials Project ID of the benchmarking structure.

        """
        flows = []
        fit_input = {}
        joined_data = {}
        collect = []

        if xyz_file is None:
            raise Exception("Error. Please provide an existing xyz file.")

        for i, structure in enumerate(structure_list):
            mp_id = mp_ids[i]
            if self.add_dft_random_struct:
                addDFTrand = self.add_dft_random(
                    structure,
                    mp_ids[i],
                    self.phonon_displacement_maker,
                    self.n_struct,
                    self.sc,
                    self.supercell_matrix,
                )
                flows.append(addDFTrand)
                joined_data.update(addDFTrand.output)
            if self.add_dft_phonon_struct:
                addDFTphon = self.add_dft_phonons(
                    structure,
                    self.displacements,
                    self.symprec,
                    self.phonon_displacement_maker,
                    self.min_length,
                )
                flows.append(addDFTphon)
                joined_data.update(addDFTphon.output)
            if self.add_rss_struct:
                raise NotImplementedError
            fit_input.update({mp_id: joined_data})

        isoatoms = get_iso_atom(structure_list)
        flows.append(isoatoms)

        add_data_fit = PhononDFTMLFitFlow().make(
            species=isoatoms.output["species"],
            isolated_atoms_energy=isoatoms.output["energies"],
            xyz_file=xyz_file,
            fit_input=fit_input,
        )
        flows.append(add_data_fit)

        # not sure if it would make sense to put everything from here in its own flow?
        add_data_ml_phonon = get_phonon_ml_calculation_jobs(
            structure=benchmark_structure,
            min_length=self.min_length,
            ml_dir=add_data_fit.output,
        )
        flows.append(add_data_ml_phonon)
        if mp_id not in mp_ids or self.add_dft_phonon_struct is False:
            dft_phonons = DFTPhononMaker(
                symprec=self.symprec,
                phonon_displacement_maker=self.phonon_displacement_maker,
                born_maker=None,
                min_length=self.min_length,
            ).make(structure=benchmark_structure)
            dft_phonons = update_user_incar_settings(
                dft_phonons, {"NPAR": 4, "ISPIN": 1}
            )

            flows.append(dft_phonons)

            dft_reference = dft_phonons.output
        else:
            dft_reference = fit_input[mp_id]["phonon_data"]["001"]

        add_data_bm = PhononDFTMLBenchmarkFlow(name="addDataBM").make(
            structure=benchmark_structure,
            mp_id=mp_id,
            ml_phonon_task_doc=add_data_ml_phonon.output,
            dft_phonon_task_doc=dft_reference,
        )
        flows.append(add_data_bm)
        collect.append(add_data_bm.output)

        collect_bm = write_benchmark_metrics(
            benchmark_structure=benchmark_structure,
            mp_id=mp_id,
            rmse=collect,
            displacements=self.displacements,
        )
        flows.append(collect_bm)

        return Flow(flows, collect_bm.output)

    def add_dft_phonons(
        self,
        structure: Structure,
        displacements,
        symprec,
        phonon_displacement_maker,
        min_length,
    ):
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
        mp_id,
        phonon_displacement_maker,
        n_struct,
        sc,
        supercell_matrix: Matrix3D | None = None,
    ):
        additonal_dft_random = dft_random_gen_data(
            structure, mp_id, phonon_displacement_maker, n_struct, sc, supercell_matrix
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
    sc: bool.
        If True, will generate randomly distorted supercells structures
        and add static computation jobs to the flow
    """

    name: str = "datagen"
    phonon_displacement_maker: BaseVaspMaker = field(
        default_factory=PhononDisplacementMaker
    )
    n_struct: int = 1
    displacements: list[float] = field(default_factory=lambda: [0.01])
    min_length: int = 20
    symprec: float = 1e-4
    sc: bool = False
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
        dft_random = dft_random_gen_data(
            structure,
            mp_id,
            self.phonon_displacement_maker,
            self.n_struct,
            self.sc,
            self.supercell_matrix,
        )

        return Flow(
            [dft_phonon, dft_random],  # flows
            output={
                "rand_struc_dir": dft_random.output,
                "phonon_dir": dft_phonon.output["dirs"],
                "phonon_data": dft_phonon.output["data"],
            },
        )


@dataclass
class PhononDFTMLFitFlow(Maker):
    """
    Maker to fit ML potentials based on DFT data.

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
            Mixed list of dictionary and lists
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
        mp_id: str,
        ml_phonon_task_doc: PhononBSDOSDoc,
        dft_phonon_task_doc: PhononBSDOSDoc,
    ):
        """
        Create flow to benchmark the ML potential.

        Parameters
        ----------
        structure: Structure
            Structure used for benchmark
        mp_id: str.
            Material project id string
        ml_phonon_task_doc: PhononBSDOSDoc
            Phonon task doc from ML potential consisting of pymatgen band-structure object
        dft_phonon_task_doc: PhononBSDOSDoc
            Phonon task doc from DFT runs consisting of pymatgen band-structure object
        """
        flows = []

        benchmark = PhononBenchmarkMaker(name="Benchmark").make(
            structure=structure,
            mp_id=mp_id,
            ml_phonon_bs=ml_phonon_task_doc.phonon_bandstructure,  # TODO take BS at top lvl
            dft_phonon_bs=dft_phonon_task_doc.phonon_bandstructure,
        )
        flows.append(benchmark)

        return Flow(flows, benchmark.output)
