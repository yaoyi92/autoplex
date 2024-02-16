"""The mlip_fitting part."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import ase.io
from jobflow import Maker, job

from autoplex.fitting.common.utils import (
    get_list_of_vasp_calc_dirs,
    outcar_2_extended_xyz,
)
from autoplex.fitting.common.jobs import gap_fitting  # , ace_fitting
from autoplex.fitting.common.regularization import set_sigma
from autoplex.fitting.common.utils import data_distillation, split_dataset


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
        pre_database_dir: str,
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
        config_types = []

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

        # database_path = Path.cwd()

        return Path.cwd()  # database_path


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
        # nequip: dict,
        gap_para=None,
        # ace_para={
        #     "energy_name": "REF_energy",
        #     "force_name": "REF_forces",
        #     "virial_name": "REF_virials",
        #     "order": 3,
        #     "totaldegree": 6,
        #     "cutoff": 2.0,
        #     "solver": "BLR",
        # },
        isol_es: None = None,
        num_of_threads: int = 128,
        **kwargs,
    ):
        """
        Maker for data preprocessing.

        Parameters
        ----------
        database_dir:
            the database directory.
        nequip:
            nequip parameters.
        gap_para: dict
            gap fit parameters.
        isol_es:
            isolated es.
        num_of_threads: int
            number of threads to be used.

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
            train_error, test_error = gap_fitting(
                dir=database_dir,
                two_body=gap_para["two_body"],
                three_body=gap_para["three_body"],
                soap=["soap"],
            )

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
