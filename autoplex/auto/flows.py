"""Flows perform automatic data generation, fitting, and benchmarking of ML potentials."""

from dataclasses import dataclass, field

from atomate2.common.jobs.phonons import PhononDisplacementMaker
from atomate2.common.schemas.phonons import PhononBSDOSDoc
from atomate2.vasp.flows.phonons import PhononMaker as DFTPhononMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.powerups import (
    update_user_incar_settings,
)
from jobflow import Flow, Maker
from pymatgen.core.structure import Structure

from autoplex.auto.jobs import get_phonon_ml_calculation_jobs
from autoplex.benchmark.flows import PhononBenchmarkMaker
from autoplex.benchmark.jobs import write_benchmark_metrics
from autoplex.data.flows import DataGenerator, IsoAtomMaker
from autoplex.fitting.flows import MLIPFitMaker

__all__ = [
    "CompleteDFTvsMLBenchmarkWorkflow",
    "PhononDFTMLDataGenerationFlow",
    "PhononDFTMLBenchmarkFlow",
]


# Volker's idea: provide several default flows with different setting/setups


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
        If True, will generate supercells of initial randomly displaced
        structures and add phonon computation jobs to the flow

    """

    name: str = "complete_workflow"
    n_struct: int = 1
    displacements: list[float] = field(default_factory=lambda: [0.01])
    min_length: int = 20
    symprec: float = 1e-4
    sc: bool = False

    def make(
        self,
        structure_list: list[Structure],
        mpids,
        phonon_displacement_maker,
        benchmark_structure: Structure,
        mp_id,
    ):
        """

        Parameters
        ----------
        structure_list: List[Structure]
            list of pymatgen structures
        mpids : list.
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
        isoatoms = []
        all_species = list(set([specie for s in structure_list for specie in s.types_of_species]))

        for species in all_species:
            isoatom = IsoAtomMaker().make(species=species)
            flows.append(isoatom)
            isoatoms.append(isoatom.output)

        for struc_i, structure in enumerate(structure_list):
            mp_id = mpids[struc_i]
            autoplex_datagen = PhononDFTMLDataGenerationFlow(
                name="test",
                phonon_displacement_maker=phonon_displacement_maker,
                n_struct=self.n_struct,
                displacements=self.displacements,
                min_length=self.min_length,
                symprec=self.symprec,
                sc=self.sc,
            ).make(structure=structure, mp_id=mp_id)
            flows.append(autoplex_datagen)
            datagen.update({mp_id: autoplex_datagen.output})

        autoplex_fit = PhononDFTMLFitFlow().make(
            species=all_species,
            isolated_atoms_energy=isoatoms,
            fit_input=datagen,
        )
        flows.append(autoplex_fit)

        # for struc_i, structure in enumerate(structure_list):  ### just commenting this out for now
        autoplex_ml_phonon = get_phonon_ml_calculation_jobs(
            structure=benchmark_structure,
            min_length=self.min_length,
            ml_dir=autoplex_fit.output,
        )
        flows.append(autoplex_ml_phonon)
        if mp_id not in mpids:
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
            dft_reference = datagen[mp_id]["phonon_data"][
                0
            ]  # [0] because we only need the first entry for the displacement = 0.1

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

        flow = Flow(flows)
        return flow


@dataclass
class PhononDFTMLDataGenerationFlow(Maker):
    """
    Maker to generate DFT reference database to be used for fitting ML potentials.

    The maker will use phonopy to create displacements according to the finite displacement method.
    In addition, random displacments are applied to the provided structures.

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
        If True, will generate supercells of initial randomly displaced
        structures and add phonon computation jobs to the flow
    """

    name: str = "datagen"
    phonon_displacement_maker: BaseVaspMaker = field(
        default_factory=PhononDisplacementMaker
    )
    n_struct: int = 1
    displacements: list[float] = field(
        default_factory=lambda: [0.01]
    )  # TODO Make sure that 0.01 is always included, no matter what the user does
    min_length: int = 20
    symprec: float = 1e-4
    sc: bool = False

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
        flows = []
        dft_phonons_output = []
        dft_phonons_dir_output = []

        # TODO later adding: for i no. of potentials
        for displacement in self.displacements:
            dft_phonons = DFTPhononMaker(
                symprec=self.symprec,
                phonon_displacement_maker=self.phonon_displacement_maker,
                born_maker=None,
                displacement=displacement,
                min_length=self.min_length,
            ).make(structure=structure)
            # TODO: move this to a different level (e.g., submission script)
            dft_phonons = update_user_incar_settings(dft_phonons, {"NPAR": 4})
            flows.append(dft_phonons)
            dft_phonons_output.append(
                dft_phonons.output
            )  # I have no better solution to this now
            dft_phonons_dir_output.append(
                dft_phonons.output.jobdirs.displacements_job_dirs
            )
        datagen = DataGenerator(
            name="DataGen",
            phonon_displacement_maker=self.phonon_displacement_maker,
            n_struct=self.n_struct,
            sc=self.sc,
        ).make(structure=structure, mp_id=mp_id)
        flows.append(datagen)

        flow = Flow(
            flows,
            output={
                "rand_struc_dir": datagen.output,
                "phonon_dir": dft_phonons_dir_output,
                "phonon_data": dft_phonons_output,
            },
        )
        return flow


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

    def make(self, species, isolated_atoms_energy, fit_input, **fit_kwargs):
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
        fit_kwargs : dict.
            dict including gap fit keyword args.
        """
        flows = []

        ml_fit_flow = MLIPFitMaker(name="GAP").make(
            species_list=species,
            iso_atom_energy=isolated_atoms_energy,
            fit_input=fit_input,
            **fit_kwargs,
        )
        flows.append(ml_fit_flow)

        flow = Flow(flows, ml_fit_flow.output)
        return flow


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

        flow = Flow(flows, benchmark.output)
        return flow
