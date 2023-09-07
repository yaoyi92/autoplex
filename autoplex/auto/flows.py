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

__all__ = [
    "PhononDFTMLDataGenerationFlow",
    "PhononDFTMLBenchmarkFlow"
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
            structure_list: list[Structure],
            mpids: list,  # list[MPID]
            ml_dir: str | Path | None = None,
            **fit_kwargs
    ):
        """
        Make flow for benchmarking.

        Parameters
        ----------

        """

        flows = []
        isoatoms = []
        for species in structure_list[0].types_of_species:
            isoatom = IsoAtomMaker().make(species=species)
            flows.append(isoatom)
            isoatoms.append(isoatom.output)

        for struc_i, structure in enumerate(structure_list):  # later adding: for i no. of potentials
            DFTphonons = DFTPhononMaker(symprec = self.symprec, phonon_displacement_maker = self.phonon_displacement_maker, born_maker = None, min_length = 8).make(structure=structure) # reduced the accuracy for test calculations
            flows.append(DFTphonons)
            datagen = DataGenerator(name="DataGen", phonon_displacement_maker = self.phonon_displacement_maker).make(structure=structure, mpid=mpids[struc_i])
            flows.append(datagen)

        MLfit = MLIPFitMaker(name="GAP").make(species_list=structure_list[0].types_of_species,
                                                      iso_atom_energy=isoatoms, fitinput=DFTphonons.output.jobdirs.displacements_job_dirs, fitinputrand=datagen.output['dirs'], structurelist=structure_list, **fit_kwargs)
        flows.append(MLfit)

        flow = Flow(flows, {"ml_dir": MLfit.output, "dft_ref": DFTphonons.output}) #TODO in the future replace with separate DFT output
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
            structure_list: list[Structure],
            mpids: list,  # list[MPID]
            ml_reference,
            dft_reference,
    ):
        flows = []

        for struc_i, structure in enumerate(structure_list):
            benchmark = PhononBenchmarkMaker(name="Benchmark").make(structure=structure, mpid=mpids[struc_i],
                                                                    ml_reference=ml_reference,
                                                                    dft_reference=dft_reference)
            flows.append(benchmark)

        flow = Flow(flows)
        return flow
