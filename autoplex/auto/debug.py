"""
Total AutoPLEX -- Automated machine-learned Potential Landscape explorer -- flow
"""

from pathlib import Path
from pymatgen.core.structure import Structure
from dataclasses import dataclass, field
from jobflow import Flow, Maker, OutputReference, job
from autoplex.fitting.flows import MLIPFitMaker

__all__ = ["debugger"]

@dataclass
class debugger(Maker):

    name: str = "debug"

    def make(self, structure_list: list[Structure],):

        jobs = []
        fitinput = ['sv3006:/home/certural/Calc/block_2023-08-23-11-35-04-773632/launcher_2023-08-25-09-39-24-719380', 'sv3012:/home/certural/Calc/block_2023-08-23-11-35-04-773632/launcher_2023-08-25-09-39-19-468652'], ['sv3020:/home/certural/Calc/block_2023-08-23-11-35-04-773632/launcher_2023-08-25-09-31-48-299628', 'sv3001:/home/certural/Calc/block_2023-08-23-11-35-04-773632/launcher_2023-08-25-09-31-43-108063']

        MLfit = MLIPFitMaker(name="GAP").make(species_list=["Li", "Cl"], iso_atom_energy=[-0.28665266, -0.25639058], fitinput=fitinput, structurelist=structure_list)
        jobs.append(MLfit)

        flow = Flow(jobs)
        return flow