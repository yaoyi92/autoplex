"""Flows consisting of jobs to fit ML potentials."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import ase.io
from jobflow import Flow, Maker, job

from autoplex.fitting.common.jobs import machine_learning_fit
from autoplex.fitting.common.regularization import set_sigma
from autoplex.fitting.common.utils import (
    data_distillation,
    get_list_of_vasp_calc_dirs,
    split_dataset,
    vaspoutput_2_extended_xyz,
)

__all__ = [
    "MLIPFitMaker",
    "DataPreprocessing",
]


@dataclass
class MLIPFitMaker(Maker):
    """
    Maker to fit ML potentials based on DFT labelled reference data.

    This Maker will filter the provided dataset in a data preprocessing step and then proceed
    with the MLIP fit (default is GAP).

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    mlip_type: str
        Choose one specific MLIP type:
        'GAP' | 'SNAP' | 'ACE' | 'Nequip' | 'Allegro' | 'MACE'
    """

    name: str = "MLpotentialFit"
    mlip_type: str = "GAP"

    def make(
        self,
        species_list: list,
        isolated_atoms_energy: list,
        fit_input: dict,
        split_ratio: float = 0.4,
        f_max: float = 40.0,
        pre_xyz_files: list[str] | None = None,
        pre_database_dir: str | None = None,
        regularization: float = 0.1,
        f_min: float = 0.01,  # unit: eV Å-1
        atom_wise_regularization: bool = True,
        auto_delta: bool = True,
        glue_xml: bool = False,
        **fit_kwargs,
    ):
        """
        Make a flow to create ML potential fits.

        Parameters
        ----------
        species_list : list.
            List of element names (str)
        isolated_atoms_energy : list.
            List of isolated atoms energy
        fit_input : dict.
            PhononDFTMLDataGenerationFlow output
        split_ratio: float.
            Parameter to divide the training set and the test set.
            A value of 0.1 means that the ratio of the training set to the test set is 9:1.
        f_max: float
            Maximally allowed force in the data set.
        pre_xyz_files: list[str] or None
            names of the pre-database train xyz file and test xyz file.
        pre_database_dir:
            the pre-database directory.
        regularization: float
            regularization value for the atom-wise force components.
        f_min: float
            minimal force cutoff value for atom-wise regularization.
        atom_wise_regularization: bool
            for including atom-wise regularization.
        auto_delta: bool
            automatically determine delta for 2b, 3b and soap terms.
        glue_xml: bool
            use the glue.xml core potential instead of fitting 2b terms.
        fit_kwargs : dict.
            dict including gap fit keyword args.
        """
        jobs = []
        data_prep_job = DataPreprocessing(
            split_ratio=split_ratio, regularization=True, distillation=True, f_max=f_max
        ).make(
            fit_input=fit_input,
            pre_xyz_files=pre_xyz_files,
            pre_database_dir=pre_database_dir,
            f_min=f_min,
            regularization=regularization,
            atom_wise_regularization=atom_wise_regularization,
        )
        jobs.append(data_prep_job)
        gap_fit_job = machine_learning_fit(
            database_dir=data_prep_job.output,
            isol_es=None,
            auto_delta=auto_delta,
            glue_xml=glue_xml,
            mlip_type=self.mlip_type,
            **fit_kwargs,
        )
        jobs.append(gap_fit_job)  # type: ignore

        # create a flow including all jobs
        return Flow(jobs, gap_fit_job.output)


@dataclass
class DataPreprocessing(Maker):
    """
    Data preprocessing of the provided dataset.

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
        pre_xyz_files: list[str] | None = None,
        regularization: float = 0.1,
        f_min: float = 0.01,  # unit: eV Å-1
        atom_wise_regularization: bool = True,
    ):
        """
        Maker for data preprocessing.

        Parameters
        ----------
        fit_input:
            Mixed list of dictionary and lists of the fit input data.
        pre_database_dir: str or None
            the pre-database directory.
        pre_xyz_files: list[str] or None
            names of the pre-database train xyz file and test xyz file labeled by VASP.
        regularization: float
            regularization value for the atom-wise force components.
        f_min: float
            minimal force cutoff value for atom-wise regularization.
        atom_wise_regularization: bool
            for including atom-wise regularization.

        """
        if pre_xyz_files is None:
            pre_xyz_files = ["train.extxyz", "test.extxyz"]

        list_of_vasp_calc_dirs = get_list_of_vasp_calc_dirs(flow_output=fit_input)

        config_types = [
            key
            for key, value in fit_input.items()
            for key2, value2 in value.items()
            if key2 != "phonon_data"
            for _ in value2[0]
        ]

        data_types = [
            key2
            for key, value in fit_input.items()
            for key2, value2 in value.items()
            if key2 != "phonon_data"
            for _ in value2[0]
        ]

        if pre_database_dir and os.path.exists(pre_database_dir):
            current_working_directory = os.getcwd()

            if len(pre_xyz_files) == 1:
                for file_name in pre_xyz_files:
                    source_file_path = os.path.join(pre_database_dir, file_name)
                    destination_file_path = os.path.join(
                        current_working_directory, "vasp_ref.extxyz"
                    )
                    shutil.copy(source_file_path, destination_file_path)
                    print(
                        f"File {file_name} has been copied to {destination_file_path}"
                    )

        vaspoutput_2_extended_xyz(
            path_to_vasp_static_calcs=list_of_vasp_calc_dirs,
            config_types=config_types,
            data_types=data_types,
            f_min=f_min,
            regularization=regularization,
            atom_wise_regularization=atom_wise_regularization,
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
            current_working_directory = os.getcwd()

            if len(pre_xyz_files) == 2:
                files_new = ["train.extxyz", "test.extxyz"]
                for file_name, file_new in zip(pre_xyz_files, files_new):
                    source_file_path = os.path.join(pre_database_dir, file_name)
                    destination_file_path = os.path.join(
                        current_working_directory, file_new
                    )
                    shutil.copy(source_file_path, destination_file_path)
                    print(
                        f"File {file_name} has been copied to {destination_file_path}"
                    )
            elif len(pre_xyz_files) > 2:
                raise ValueError(
                    "Please provide a train and a test extxyz file (two files in total) for the pre_xyz_files."
                )

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
