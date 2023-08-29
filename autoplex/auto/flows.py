"""
Total AutoPLEX -- Automated machine-learned Potential Landscape explorer -- flow
"""

from pathlib import Path
from pymatgen.core.structure import Structure
from dataclasses import dataclass, field
from jobflow import Flow, Maker, OutputReference, job
from autoplex.data.flows import DataGenerator, IsoAtomMaker
from autoplex.fitting.flows import MLIPFitMaker
from autoplex.benchmark.flows import PhononBenchmarkMaker
from atomate2.forcefields.jobs import GAPRelaxMaker, GAPStaticMaker
from atomate2.forcefields.flows.phonons import PhononMaker
from atomate2.vasp.flows.phonons import PhononMaker as DFTPhononMaker
from atomate2.vasp.sets.core import StaticSetGenerator
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.common.jobs.phonons import PhononDisplacementMaker

__all__ = ["PhononDFTMLBenchmarkFlow"]


# Volker's idea: provide several default flows with different setting/setups

@dataclass
class PhononDFTMLBenchmarkFlow(Maker):
    """
    Maker to create ML potentials based on DFT data
    3. Step: Evaluate Potentials

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.

    """

    name: str = "DFTbenchmark"
    phonon_displacement_maker: BaseVaspMaker = field(default_factory=PhononDisplacementMaker)
    symprec: float = 0.01

    def make(
            self,
            structure_list: list[Structure],
            mpids: list,  # list[MPID]
            ml_dir: str | Path | None = None,
    ):
        """
        Make flow for benchmarking.

        Parameters
        ----------

        """
        # potential_names = ["GAPfit"]

        flows = []
        isoatoms = []
        fitinput = []
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
                                                      iso_atom_energy=isoatoms, fitinput=DFTphonons.output.jobdirs.displacements_job_dirs, fitinputrand=datagen.output['dirs'], structurelist=structure_list)
        flows.append(MLfit)

        #if ml_dir is None: ml_dir =

        for struc_i, structure in enumerate(structure_list):
            GAPPhonons = PhononMaker(
                bulk_relax_maker=GAPRelaxMaker(potential_param_file_name=MLfit.output["dir"], relax_cell=True,
                                               relax_kwargs={"interval": 500}),
                phonon_displacement_maker=GAPStaticMaker(potential_param_file_name=MLfit.output["dir"]),
                static_energy_maker=GAPStaticMaker(potential_param_file_name=MLfit.output["dir"]),
                store_force_constants=False,
                generate_frequencies_eigenvectors_kwargs={"units": "THz"}).make(structure=MLfit.output["struclist"][struc_i])
            flows.append(GAPPhonons)
            # benchmark = PhononBenchmarkMaker(name="Benchmark").make()
            # flows.append(benchmark)

        flow = Flow(flows)
        return flow

