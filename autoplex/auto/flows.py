"""
Total AutoPLEX -- Automated machine-learned Potential Landscape explorer -- flow
"""

from pathlib import Path
from pymatgen.core.structure import Structure
from dataclasses import dataclass, field
from jobflow import Flow, Maker, OutputReference, job
from autoplex.data.flows import DataGenerator
from autoplex.fitting.flows import MLIPFitMaker
from autoplex.benchmark.flows import PhononBenchmarkMaker

__all__ = ["PhononDFTMLBenchmarkFlow"]


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
            ml_dir: str | Path | None = None, #TODO einbauen, dass die ml_dir irgendwie verwendet wird, wenn sie vom user verwendet wird
    ):
        """
        Make flow for benchmarking..

        Parameters
        ----------

        """
        flows = []
        datagen = DataGenerator(name = "DataGen", symprec = 0.0001)
        flows.append(datagen)
        MLfit = MLIPFitMaker(name = "GAP")
        flows.append(MLfit)
        benchmark = PhononBenchmarkMaker(name = "Benchmark")
        flows.append(benchmark)

        flow = Flow(flows)
        return flow




