"""
Flows consisting of jobs to create training data for ML potentials
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from jobflow import Flow, Maker, OutputReference, job
from pymatgen.core.structure import Structure
from autoplex.data.jobs import generate_random_displacement

from atomate2.vasp.sets.core import StaticSetGenerator
from atomate2.common.jobs import structure_to_conventional, structure_to_primitive
from emmet.core.math import Matrix3D, Vector3D
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import DielectricMaker, StaticMaker, TightRelaxMaker
from atomate2.vasp.jobs.phonons import (
    PhononDisplacementMaker,
    generate_frequencies_eigenvectors,
    generate_phonon_displacements,
    get_supercell_size,
    get_total_energy_per_cell,
    run_phonon_displacements,
)

__all__ = ["DataGenerator"]


@dataclass
class DataGenerator(Maker):
    """
    Maker to create ML potentials based on DFT data
    1. Fetch Data from Materials Project and other databases (other: work in progress)
    + Perform DFT calculations (at the current point these are also used for Phonon calculations (that part
    shall be independent in the future.)

--Note:
    All phonon-related code parts have been copied from atomate2/vasp/flows/phonons.py

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.

    """

    name: str = "DataGenerationML"
    sym_reduce: bool = True
    symprec: float = 0.0001 #1e-4
    displacement: float = 0.01
    min_length: float | None = 5.0
    prefer_90_degrees: bool = True
    get_supercell_size_kwargs: dict = field(default_factory = dict)
    use_symmetrized_structure: str | None = None
    bulk_relax_maker: BaseVaspMaker | None = field(
        default_factory = lambda: DoubleRelaxMaker.from_relax_maker(TightRelaxMaker())
    )
    static_energy_maker: BaseVaspMaker | None = field(default_factory = StaticMaker)
    born_maker: BaseVaspMaker | None = field(default_factory = DielectricMaker)
    phonon_displacement_maker: BaseVaspMaker = field(default_factory = PhononDisplacementMaker)
    create_thermal_displacements: bool = True
    generate_frequencies_eigenvectors_kwargs: dict = field(default_factory = dict)
    kpath_scheme: str = "seekpath"
    code: str = "vasp"
    store_force_constants: bool = True

    def make(
            self,
            structure_list: list[Structure],
            mpids: list,  # list[MPID]
            prev_vasp_dir: str | Path | None = None,
            born: list[Matrix3D] | None = None,
            epsilon_static: Matrix3D | None = None,
            total_dft_energy_per_formula_unit: float | None = None,
            supercell_matrix: Matrix3D | None = None,
    ):
        """
        Make flow to generate the data base.

        Parameters
        ----------
        structure_list : List of structures
            List of pymatgen structures drawn from the Materials Project.
        ml_dir : str or Path or None
            ML directory to use for copying inputs/outputs.(hab noch keine Ahnung)
        supercell_matrix: Matrix of the SC
        mpids: Materials Project IDs
        """

        # Error handling (if...raise ValueError) goes here

        if self.use_symmetrized_structure not in [None, "primitive", "conventional"]:
            raise ValueError(
                "use_symmetrized_structure can only be primitive, conventional, None"
            )

        if (
                not self.use_symmetrized_structure == "primitive"
                and self.kpath_scheme != "seekpath"
        ):
            raise ValueError(
                "You can only use other kpath schemes with the primitive standard structure"
            )

        if self.kpath_scheme not in [
            "seekpath",
            "hinuma",
            "setyawan_curtarolo",
            "latimer_munro",
        ]:
            raise ValueError("kpath scheme is not implemented")

        jobs = []  # initializing empty job list
        GAPinputs: list[OutputReference] = []
        GAPisoatominput: list = []
        GAPisoatomenergyinput: list[OutputReference] = []
        PhononCollectOutput = []
        smat = []
        distance = []

        for spec in structure_list[0].types_of_species:
            iso_atom = Structure(
                lattice = [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
                species = [spec],
                coords = [[0, 0, 0]],
            )
            isoatom_calcs = StaticMaker(name = str(spec) + "-statisoatom",
                                        input_set_generator = StaticSetGenerator(
                                            user_kpoints_settings = {"grid_density": 1}, )).make(iso_atom)
            jobs.append(isoatom_calcs)
            GAPisoatominput.append(spec)
            GAPisoatomenergyinput.append(isoatom_calcs.output.output.energy_per_atom)

        for struc in structure_list:
            if self.use_symmetrized_structure == "primitive":
                # These structures are compatible with many
                # of the kpath algorithms that are used for Materials Project
                prim_job = structure_to_primitive(struc, self.symprec)
                jobs.append(prim_job)
                struc= prim_job.output
            elif self.use_symmetrized_structure == "conventional":
                # it could be beneficial to use conventional
                # standard structures to arrive faster at supercells with right
                # angles
                conv_job = structure_to_conventional(struc, self.symprec)
                jobs.append(conv_job)
                struc = conv_job.output

            if self.bulk_relax_maker is not None:
                # optionally relax the structure
                bulk = self.bulk_relax_maker.make(struc, prev_vasp_dir = prev_vasp_dir)
                jobs.append(bulk)
                struc = bulk.output.structure
                optimization_run_job_dir = bulk.output.dir_name
                optimization_run_uuid = bulk.output.uuid
            else:
                optimization_run_job_dir = None
                optimization_run_uuid = None

            # if supercell_matrix is None, supercell size will be determined
            # after relax maker to ensure that cell lengths are really larger
            # than threshold
            if supercell_matrix is None:
                supercell_job = get_supercell_size(
                    struc,
                    self.min_length,
                    self.prefer_90_degrees,
                    **self.get_supercell_size_kwargs,
                )
                jobs.append(supercell_job)
                supercell_matrix = supercell_job.output

            # get a phonon object from phonopy
            displacement_generator = generate_phonon_displacements(
                structure = struc,
                supercell_matrix = supercell_matrix,
                displacement = self.displacement,
                sym_reduce = self.sym_reduce,
                symprec = self.symprec,
                use_symmetrized_structure = self.use_symmetrized_structure,
                kpath_scheme = self.kpath_scheme,
                code = self.code,
            )
            jobs.append(displacement_generator)

            random_rattle_displacement = generate_random_displacement(
                displacements = displacement_generator.output,
            )
            jobs.append(random_rattle_displacement)

            # perform the phonon displacement calculations
            vasp_displacement_calcs = run_phonon_displacements(
                displacements = displacement_generator.output,
                structure = struc,
                supercell_matrix = supercell_matrix,
                phonon_maker = self.phonon_displacement_maker,
            )
            jobs.append(vasp_displacement_calcs)

            # perform the phonon displacement calculations for randomized displaced structures
            vasp_random_displacement_calcs = run_phonon_displacements(
                displacements = random_rattle_displacement.output,
                structure = struc,
                supercell_matrix = supercell_matrix,
                phonon_maker = self.phonon_displacement_maker,
            )
            jobs.append(vasp_random_displacement_calcs)

            GAPinputs.append(vasp_displacement_calcs.output)  # collect displaced structures, energies, forces for fit
            GAPinputs.append(vasp_random_displacement_calcs.output)
            distance.append(self.displacement)

            # Computation of static energy
            if (self.static_energy_maker is not None) and (
                    total_dft_energy_per_formula_unit is None
            ):
                static_job = self.static_energy_maker.make(structure = struc)
                jobs.append(static_job)
                total_dft_energy = static_job.output.output.energy
                static_run_job_dir = static_job.output.dir_name
                static_run_uuid = static_job.output.uuid
            else:
                if total_dft_energy_per_formula_unit is not None:
                    # to make sure that one can reuse results from Doc
                    compute_total_energy_job = get_total_energy_per_cell(
                        total_dft_energy_per_formula_unit, struc
                    )
                    jobs.append(compute_total_energy_job)
                    total_dft_energy = compute_total_energy_job.output
                else:
                    total_dft_energy = None
                static_run_job_dir = None
                static_run_uuid = None

            # Computation of BORN charges
            if self.born_maker is not None and (born is None or epsilon_static is None):
                born_job = self.born_maker.make(struc)
                jobs.append(born_job)

                # I am not happy how we currently access "born" charges
                # This is very vasp specific code
                epsilon_static = born_job.output.calcs_reversed[0].output.epsilon_static
                born = born_job.output.calcs_reversed[0].output.outcar["born"]
                born_run_job_dir = born_job.output.dir_name
                born_run_uuid = born_job.output.uuid
            else:
                born_run_job_dir = None
                born_run_uuid = None

            phonon_collect = generate_frequencies_eigenvectors(
                supercell_matrix = supercell_matrix,
                displacement = self.displacement,
                sym_reduce = self.sym_reduce,
                symprec = self.symprec,
                use_symmetrized_structure = self.use_symmetrized_structure,
                kpath_scheme = self.kpath_scheme,
                code = self.code,
                structure = struc,
                displacement_data = vasp_displacement_calcs.output,
                epsilon_static = epsilon_static,
                born = born,
                total_dft_energy = total_dft_energy,
                static_run_job_dir = static_run_job_dir,
                static_run_uuid = static_run_uuid,
                born_run_job_dir = born_run_job_dir,
                born_run_uuid = born_run_uuid,
                optimization_run_job_dir = optimization_run_job_dir,
                optimization_run_uuid = optimization_run_uuid,
                create_thermal_displacements = self.create_thermal_displacements,
                store_force_constants = self.store_force_constants,
                **self.generate_frequencies_eigenvectors_kwargs,
            )
            jobs.append(phonon_collect)

            PhononCollectOutput.append(phonon_collect.output.phonon_bandstructure)

            smat.append(supercell_matrix)

            # set these parameters to "None" to trigger the calculation for each structure in structure_list
            supercell_matrix = None
            born = None
            epsilon_static = None

        # create a flow including all jobs
        flow = Flow(jobs, PhononCollectOutput)
        return flow
