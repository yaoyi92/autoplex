"""
PLEASE IGNORE ANY FILES CONCERNING DEBUGGING
"""

from pathlib import Path
from pymatgen.core.structure import Structure
from dataclasses import dataclass, field
from jobflow import Flow, Maker, OutputReference, job
from autoplex.fitting.flows import MLIPFitMaker
from autoplex.benchmark.flows import PhononBenchmarkMaker

__all__ = ["debugger"]

@dataclass
class debugger(Maker):

    name: str = "debug"

    def make(self, structure_list: list[Structure], **fit_kwargs):

        jobs = []
        fitinput = ['sv2002:/home/certural/Calc/block_2023-08-22-07-49-56-741130/launcher_2023-08-30-12-27-05-899191', 'sv2001:/home/certural/Calc/block_2023-08-22-07-49-56-741130/launcher_2023-08-30-12-27-00-711295']
        fitinputrand = ['sv2005:/home/certural/Calc/block_2023-08-22-07-49-56-741130/launcher_2023-08-30-12-00-13-957930']

        MLfit = MLIPFitMaker(name="GAP").make(species_list=["Li", "Cl"], iso_atom_energy=[-0.28665266, -0.25639058], fitinput=fitinput, fitinputrand = fitinputrand, **fit_kwargs)
        jobs.append(MLfit)

        benchmark = PhononBenchmarkMaker(name="Benchmark").make(structure_list=structure_list, mpids=[])
        jobs.append(benchmark)

        flow = Flow(jobs, MLfit.output)
        return flow

