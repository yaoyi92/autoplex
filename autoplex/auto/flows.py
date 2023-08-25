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

    def make(
            self,
            structure_list: list[Structure],
            mpids: list,  # list[MPID]
            ml_dir: str | Path | None = None,
    ):
        """
        Make flow for benchmarking..

        Parameters
        ----------

        """
        # potential_names = ["GAPfit"]

        flows = []
        isoatoms = []
        fitinput = []
        phonon_stat = BaseVaspMaker(input_set_generator = StaticSetGenerator(user_kpoints_settings = {"grid_density": 1}, )) # reduced the accuracy for test calculations
        for species in structure_list[0].types_of_species:
            isoatom = IsoAtomMaker().make(species=species)
            flows.append(isoatom)
            isoatoms.append(isoatom.output)

        for struc_i, structure in enumerate(structure_list):  # later adding: for i no. of potentials
            DFTphonons = DFTPhononMaker(symprec = 0.01, phonon_displacement_maker = phonon_stat, born_maker = None, min_length = 8).make(structure=structure) # reduced the accuracy for test calculations
            flows.append(DFTphonons)
            fitinput.append({"dft": DFTphonons.output.jobdirs.displacements_job_dirs})
            datagen = DataGenerator(name="DataGen").make(structure=structure, mpid=mpids[struc_i])
            flows.append(datagen)
            fitinput.append({"random": datagen.output['dirs']})

        MLfit = MLIPFitMaker(name="GAP").make(species_list=structure_list[0].types_of_species,
                                                      iso_atom_energy=isoatoms, fitinput=fitinput, structurelist=structure_list)
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

@dataclass
class debugger(Maker):

    name: str = "debug"

    def make(self, structure_list: list[Structure],):

        jobs = []
        fitinput = {'dft': ['sv3006:/home/certural/Calc/block_2023-08-23-11-35-04-773632/launcher_2023-08-25-09-39-24-719380', 'sv3012:/home/certural/Calc/block_2023-08-23-11-35-04-773632/launcher_2023-08-25-09-39-19-468652'], 'random': ['sv3020:/home/certural/Calc/block_2023-08-23-11-35-04-773632/launcher_2023-08-25-09-31-48-299628', 'sv3001:/home/certural/Calc/block_2023-08-23-11-35-04-773632/launcher_2023-08-25-09-31-43-108063']}

        MLfit = MLIPFitMaker(name="GAP").make(species_list=["Li", "Cl"], iso_atom_energy=[-0.28665266, -0.25639058], fitinput=fitinput, structurelist=structure_list)
        jobs.append(MLfit)

        flow = Flow(jobs)
        return flow