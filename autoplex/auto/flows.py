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
from atomate2.vasp.powerups import update_user_incar_settings, update_user_kpoints_settings
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

    name: str = "datagen"
    phonon_displacement_maker: BaseVaspMaker = field(default_factory=PhononDisplacementMaker)
    n_struc: int = 1
    displacements: list[float] = field(default_factory=lambda: [0.1])
    min_length: int = 20
    symprec: float = 0.01
    sc: bool = False

    def make(
            self,
            structure: Structure,
            mpid
    ):
        """
        Make flow for data generation.

        Parameters
        ----------

        """
        flows = []
        DFTphonons_output = []
        DFTphonons_dir_output = []

        # TODO later adding: for i no. of potentials
        for displacement in self.displacements:
            DFTphonons = DFTPhononMaker(symprec=self.symprec,
                                        phonon_displacement_maker=self.phonon_displacement_maker, born_maker=None,
                                        displacement=displacement, min_length=self.min_length).make(structure=structure)
            DFTphonons = update_user_incar_settings(DFTphonons, {"NPAR": 4})
            flows.append(DFTphonons)
            DFTphonons_output.append(DFTphonons.output) # I have no better solution to this now
            DFTphonons_dir_output.append(DFTphonons.output.jobdirs.displacements_job_dirs)
        datagen = DataGenerator(name="DataGen",
                                phonon_displacement_maker=self.phonon_displacement_maker,
                                n_struc=self.n_struc, sc=self.sc).make(structure=structure, mpid=mpid)
        flows.append(datagen)

        flow = Flow(flows, {"rand_struc_dir": datagen.output, "phonon_dir": DFTphonons_dir_output,
                            "phonon_data": DFTphonons_output}) # TODO in the future: DFT for fit and benchmark doesn't have to be the same
        return flow

@dataclass
class PhononDFTMLFitFlow(Maker):
    """
    Maker to fit ML potentials based on DFT data

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.

    """

    name: str = "ML fit"

    def make(
            self,
            species,
            isolated_atoms,
            fitinput,
            **fit_kwargs
    ):
        """
        Make flow for ML fit.

        Parameters
        ----------

        """
        flows = []

        MLfit = MLIPFitMaker(name="GAP").make(species_list=species, iso_atom_energy=isolated_atoms,
                                              fitinput=fitinput, **fit_kwargs)
        flows.append(MLfit)

        flow = Flow(flows, MLfit.output)
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
    n_struc: int = 1
    displacements: list[float] = field(default_factory=lambda: [0.1])
    min_length: int = 20
    symprec: float = 0.01
    sc: bool = False

    def make(
            self,
            structure_list: list[Structure],
            mpids,
            phonon_displacement_maker
    ):
        flows = []
        datagen = []
        collect = []
        isoatoms = []
        all_species = set([s.types_of_species for s in structure_list])
        for species in next(iter(all_species)):
            isoatom = IsoAtomMaker().make(species=species)
            flows.append(isoatom)
            isoatoms.append(isoatom.output)

        for struc_i, structure in enumerate(structure_list):
            autoplex_datagen = PhononDFTMLDataGenerationFlow(name="test",
                                                             phonon_displacement_maker=phonon_displacement_maker,
                                                             n_struc=self.n_struc, displacements=self.displacements,
                                                             min_length=self.min_length, symprec=self.symprec,
                                                             sc=self.sc).make(structure=structure, mpid=mpids[struc_i])
            flows.append(autoplex_datagen)
            datagen.append(autoplex_datagen.output)

        autoplex_fit = PhononDFTMLFitFlow().make(species=next(iter(all_species)), isolated_atoms=isoatoms,
                                                 fitinput=datagen)
        flows.append(autoplex_fit)

        for struc_i, structure in enumerate(structure_list):
            autoplex_ml_phonon = PhononMLCalculationJob(structure=structure, displacements=self.displacements,
                                                        min_length=self.min_length, ml_dir=autoplex_fit.output)
            flows.append(autoplex_ml_phonon)
            autoplex_bm = PhononDFTMLBenchmarkFlow(name="testBM").make(structure=structure, mpid=mpids[struc_i],
                                                                       ml_reference=autoplex_ml_phonon.output,
                                                                       dft_reference=datagen[struc_i]["phonon_data"])
            flows.append(autoplex_bm)
            collect.append(autoplex_bm.output)

        collect_bm = CollectBenchmark(structure_list=structure_list, mpids=mpids, rms_dis=collect,
                                      displacements=self.displacements)
        flows.append(collect_bm)

        flow = Flow(flows)
        return flow
