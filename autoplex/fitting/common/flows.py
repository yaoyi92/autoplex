"""Flows consisting of jobs to fit ML potentials."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import ase.io
from jobflow import Flow, Maker, job

from autoplex.fitting.common.jobs import gap_fitting
from autoplex.fitting.common.regularization import set_sigma
from autoplex.fitting.common.utils import (
    data_distillation,
    get_list_of_vasp_calc_dirs,
    outcar_2_extended_xyz,
    split_dataset,
)

__all__ = [
    "CompleteMLIPFitMaker",
    "DataPreprocessing",
    "MLIPFitMaker",
]


@dataclass
class CompleteMLIPFitMaker(Maker):
    """
    Maker to fit ML potentials based on DFT data.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    """

    name: str = "CompleteMLpotentialFit"

    def make(
        self,
        species_list: list,
        iso_atom_energy: list,
        fit_input: dict,
        split_ratio: float = 0.4,
        f_max: float = 40.0,
        xyz_file: str | None = None,
        pre_database_dir: str | None = None,
        auto_delta: bool = True,
        **fit_kwargs,
    ):
        """
        Make flow to create ML potential fits.

        Parameters
        ----------
        species_list : list.
            List of element names (str)
        iso_atom_energy : list.
            List of isolated atoms energy
        fit_input : dict.
            PhononDFTMLDataGenerationFlow output
        split_ratio: float.
            Parameter to divide the training set and the test set.
            A value of 0.1 means that the ratio of the training set to the test set is 9:1.
        f_max: float
            Maximally allowed force in the data set.
        xyz_file: str or None
            a possibly already existing xyz file
        pre_database_dir:
            the pre-database directory.
        auto_delta: bool
            automatically determine delta for 2b, 3b and soap terms.
        fit_kwargs : dict.
            dict including gap fit keyword args.
        """
        jobs = []
        data_prep_job = DataPreprocessing(
            split_ratio=split_ratio, regularization=True, distillation=True, f_max=f_max
        ).make(
            fit_input=fit_input, xyz_file=xyz_file, pre_database_dir=pre_database_dir
        )
        jobs.append(data_prep_job)
        gap_fit_job = MLIPFitMaker(mlip_type="GAP").make(
            database_dir=data_prep_job.output,
            isol_es=None,
            auto_delta=auto_delta,
            **fit_kwargs,
        )
        jobs.append(gap_fit_job)  # type: ignore

        # create a flow including all jobs
        return Flow(jobs, gap_fit_job.output)


@dataclass
class DataPreprocessing(Maker):
    """
    Data preprocessing function.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    split_ratio: float
        Parameter to divide the training set and the test set.
        A value of 0.1 means that the ratio of the training set to the test set is 9:1
    regularization: bool
        For using regularization.
    distillation: bool
        For using distillation.
    f_max: float
        Maximally allowed force in the data set.

    """

    name: str = "data_preprocessing_for_fitting"
    split_ratio: float = 0.5
    regularization: bool = False
    distillation: bool = False
    f_max: float = 40.0

    @job
    def make(
        self,
        fit_input: dict,
        pre_database_dir: str | None = None,
        xyz_file: str | None = None,
    ):
        """
        Maker for data preprocessing.

        Parameters
        ----------
        fit_input:
            Mixed list of dictionary and lists of the fit input data.
        pre_database_dir:
            the pre-database directory.
        xyz_file:
            the already existing training datasets labeled by VASP.

        """
        list_of_vasp_calc_dirs = get_list_of_vasp_calc_dirs(flow_output=fit_input)

        config_types = [
            key
            for key, value in fit_input.items()
            for key2, value2 in value.items()
            if key2 != "phonon_data"
            for _ in value2[0]
        ]

        outcar_2_extended_xyz(
            path_to_vasp_static_calcs=list_of_vasp_calc_dirs,
            config_types=config_types,
            xyz_file=xyz_file,
        )

        # reject structures with large force components
        atoms = (
            data_distillation("vasp_ref.extxyz", self.f_max)
            if self.distillation
            else ase.io.read("vasp_ref.extxyz", index=":")
        )

        # split dataset into training and testing datasets with a ratio of 9:1
        (train_structures, test_structures) = split_dataset(atoms, self.split_ratio)

        # Merging database
        if pre_database_dir and os.path.exists(pre_database_dir):
            files_to_copy = ["train.extxyz", "test.extxyz"]
            current_working_directory = os.getcwd()

            for file_name in files_to_copy:
                source_file_path = os.path.join(pre_database_dir, file_name)
                destination_file_path = os.path.join(
                    current_working_directory, file_name
                )
                shutil.copy(source_file_path, destination_file_path)
                print(f"File {file_name} has been copied to {destination_file_path}")

        ase.io.write("train.extxyz", train_structures, format="extxyz", append=True)
        ase.io.write("test.extxyz", test_structures, format="extxyz", append=True)

        if self.regularization:
            atoms = ase.io.read("train.extxyz", index=":")
            atom_with_sigma = set_sigma(
                atoms,
                etup=[(0.1, 1), (0.001, 0.1), (0.0316, 0.316), (0.0632, 0.632)],
            )
            ase.io.write("train_with_sigma.extxyz", atom_with_sigma, format="extxyz")

        return Path.cwd()


@dataclass
class MLIPFitMaker(Maker):
    """
    Maker to fitting potential.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    mlip_type: str
        Choose one specific MLIP type:
        'GAP' | 'SNAP' | 'ACE' | 'Nequip' | 'Allegro' | 'MACE'
    HPO: bool
        call hyperparameter optimization (HPO) or not

    """

    name: str = "MLIP_FIT"
    mlip_type: str | None = None
    HPO: bool = False

    @job
    def make(
        self,
        database_dir: str,
        gap_para=None,
        isol_es: None = None,
        num_processes: int = 32,
        auto_delta: bool = True,
        **kwargs,
    ):
        """
        Maker for data preprocessing.

        Parameters
        ----------
        database_dir:
            the database directory.
        gap_para: dict
            gap fit parameters.
        isol_es:
            isolated es.
        num_processes: int
            number of processes for fitting.
        auto_delta: bool
            automatically determine delta for 2b, 3b and soap terms.
        kwargs: dict.
            optional dictionary with parameters for gap fitting.
        """
        if gap_para is None:
            gap_para = {"two_body": True, "three_body": False, "soap": True}

        mlip_path = Path.cwd()
        if os.path.join(database_dir, "train_with_sigma.extxyz"):
            shutil.copy(
                os.path.join(database_dir, "train_with_sigma.extxyz"),
                os.path.join(mlip_path, "train_with_sigma.extxyz"),
            )
        shutil.copy(
            os.path.join(database_dir, "test.extxyz"),
            os.path.join(mlip_path, "test.extxyz"),
        )
        shutil.copy(
            os.path.join(database_dir, "train.extxyz"),
            os.path.join(mlip_path, "train.extxyz"),
        )

        if self.mlip_type is None:
            raise ValueError(
                "MLIP type is not defined! "
                "The current version supports the fitting of GAP, SNAP, ACE, Nequip, Allegro, or MACE."
            )

        if self.mlip_type == "GAP":
            train_test_error = gap_fitting(
                db_dir=database_dir,
                include_two_body=gap_para["two_body"],
                include_three_body=gap_para["three_body"],
                include_soap=gap_para["soap"],
                num_processes=num_processes,
                auto_delta=auto_delta,
                fit_kwargs=kwargs,
            )

            train_error = train_test_error["train_error"]
            test_error = train_test_error["test_error"]

        convergence = False
        if test_error < 0.01:
            convergence = True

        return {
            "mlip_path": mlip_path,
            "mlip_xml": mlip_path.joinpath("gap_file.xml"),
            "train_error": train_error,
            "test_error": test_error,
            "convergence": convergence,
        }
