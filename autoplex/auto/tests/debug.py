"""
PLEASE IGNORE ANY FILES CONCERNING DEBUGGING
"""

from pathlib import Path
from pymatgen.core.structure import Structure
from dataclasses import dataclass, field
from jobflow import Flow, Maker, OutputReference, job
from autoplex.auto.jobs import PhononMLCalculationJob

__all__ = ["debugger"]

@dataclass
class debugger(Maker):

    name: str = "debug"

    def make(self, structure_list: list[Structure], **fit_kwargs):

        jobs = []

        for struc in structure_list:
            debug = PhononMLCalculationJob(structure=struc, displacements=[0.1], min_length=20, ml_dir="/home/certural/Calc/testAtomateTwo/YuxingPot/gap.xml")
            #debug = PhononMLCalculationJob(structure=struc, displacements=[0.1], min_length=20, ml_dir="/home/certural/Calc/block_2023-09-24-23-23-10-568937/launcher_2023-09-25-08-43-25-183334/gap.xml")
            jobs.append(debug)

        flow = Flow(jobs)
        return flow

