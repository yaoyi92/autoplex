"""
Complete AutoPLEX -- Automated machine-learned Potential Landscape explorer -- flows
"""

from pathlib import Path
from pymatgen.core.structure import Structure
from dataclasses import dataclass, field
from jobflow import Flow, Maker
from autoplex.data.flows import DataGenerator, IsoAtomMaker
from autoplex.fitting.flows import MLIPFitMaker
from atomate2.vasp.flows.phonons import PhononMaker as DFTPhononMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.common.jobs.phonons import PhononDisplacementMaker

from autoplex.benchmark.flows import PhononBenchmarkMaker
from autoplex.auto.jobs import CollectBenchmark, PhononMLCalculationJob

__all__ = [
    "PhononDFTMLDataGenerationFlow",
    "PhononDFTMLBenchmarkFlow",
    "CompleteWorkflow"
]


# Volker's idea: provide several default flows with different setting/setups

@dataclass
class PhononDFTMLDataGenerationFlow(Maker):
    """
    Maker to fit ML potentials based on DFT data

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.

    """

    name: str = "datagen+fit"
    phonon_displacement_maker: BaseVaspMaker = field(default_factory=PhononDisplacementMaker)
    symprec: float = 0.01

    def make(
            self,
            structure: Structure,
            mpid,
            isolated_atoms,
            ml_dir: str | Path | None = None,
            **fit_kwargs
    ):
        """
        Make flow for benchmarking.

        Parameters
        ----------

        """

        flows = []

        # TODO later adding: for i no. of potentials
        DFTphonons = DFTPhononMaker(symprec=self.symprec,
                                    phonon_displacement_maker=self.phonon_displacement_maker, born_maker=None,
                                    min_length=8).make(
            structure=structure)  # reduced the accuracy for test calculations
        flows.append(DFTphonons)
        datagen = DataGenerator(name="DataGen",
                                phonon_displacement_maker=self.phonon_displacement_maker).make(structure=structure,
                                                                                               mpid=mpid)
        flows.append(datagen)

        MLfit = MLIPFitMaker(name="GAP").make(species_list=structure.types_of_species, iso_atom_energy=isolated_atoms,
                                              fitinput=DFTphonons.output.jobdirs.displacements_job_dirs,
                                              fitinputrand=datagen.output['dirs'], **fit_kwargs)
        flows.append(MLfit)

        flow = Flow(flows, {"ml_dir": MLfit.output,
                            "dft_ref": DFTphonons.output})  # TODO in the future replace with separate DFT output
        return flow


@dataclass
class PhononDFTMLBenchmarkFlow(Maker):
    """
    Maker for benchmarking

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.

    """

    name: str = "MLDFTbenchmark"

    def make(
            self,
            structure: Structure,
            mpid,
            ml_reference,
            dft_reference,
    ):
        flows = []

        benchmark = PhononBenchmarkMaker(name="Benchmark").make(structure=structure, mpid=mpid,
                                                                ml_reference=ml_reference,
                                                                dft_reference=dft_reference)
        flows.append(benchmark)

        flow = Flow(flows, benchmark.output)
        return flow


@dataclass
class CompleteWorkflow(Maker):
    """
    Maker for benchmarking

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.

    """

    name: str = "complete_workflow"

    def make(
            self,
            structure_list: list[Structure],
            mpids,
            phonon_displacement_maker
    ):
        flows = []
        collect = []
        isoatoms = []
        for species in structure_list[0].types_of_species:
            isoatom = IsoAtomMaker().make(species=species)
            flows.append(isoatom)
            isoatoms.append(isoatom.output)

        for struc_i, structure in enumerate(structure_list):
            autoplex_datagen = PhononDFTMLDataGenerationFlow(name="test",
                                                             phonon_displacement_maker=phonon_displacement_maker).make(
                structure=structure, mpid=mpids[struc_i], isolated_atoms=isoatoms)
            flows.append(autoplex_datagen)
            autoplex_ml_phonon = PhononMLCalculationJob(structure=structure,
                                                        ml_dir=autoplex_datagen.output["ml_dir"])
            flows.append(autoplex_ml_phonon)
            autoplex_bm = PhononDFTMLBenchmarkFlow(name="testBM").make(structure=structure, mpid=mpids[struc_i],
                                                                       ml_reference=autoplex_ml_phonon.output,
                                                                       dft_reference=autoplex_datagen.output["dft_ref"])
            flows.append(autoplex_bm)
            collect.append(autoplex_bm.output)

        collect_bm = CollectBenchmark(structure_list=structure_list, mpids=mpids, rms=collect)
        flows.append(collect_bm)

        flow = Flow(flows)
        return flow
