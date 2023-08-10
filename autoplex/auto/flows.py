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

__all__ = ["PhononDFTMLBenchmarkFlow"]


# Idee von Volker: verschiedene vorgefertigte flows mit verschiedenen Einstellungen bereit stellen

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
        for species in structure_list[0].types_of_species:
            isoatom = IsoAtomMaker().make(species=species)
            flows.append(isoatom)
            isoatoms.append(isoatom.output)

        for struc_i, structure in enumerate(structure_list):  # later adding: for i no. of potentials
            DFTphonons = DFTPhononMaker().make(structure=structure)
            flows.append(DFTphonons)
            datagen = DataGenerator(name="DataGen", symprec=0.0001).make(structure=structure, mpid=mpids[struc_i])
            flows.append(datagen)

            MLfit = MLIPFitMaker(name="GAP").make(species_list=structure_list[0].types_of_species,
                                                      iso_atom_energy=isoatoms, phonon_structures=DFTphonons.output,
                                                      rattled_structures=datagen.output)
            flows.append(MLfit)
            #if ml_dir is None: ml_dir =

            GAPPhonons = PhononMaker(
                bulk_relax_maker=GAPRelaxMaker(potential_param_file_name=MLfit.output, relax_cell=True,
                                               relax_kwargs={"interval": 500}),
                phonon_displacement_maker=GAPStaticMaker(potential_param_file_name=MLfit.output),
                static_energy_maker=GAPStaticMaker(potential_param_file_name=MLfit.output),
                store_force_constants=False,
                generate_frequencies_eigenvectors_kwargs={"units": "THz"}).make(structure=structure)
            flows.append(GAPPhonons)
            # benchmark = PhononBenchmarkMaker(name="Benchmark").make()
            # flows.append(benchmark)

        flow = Flow(flows)
        return flow
