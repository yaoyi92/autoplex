from __future__ import annotations

from pymatgen.core.structure import Structure
from autoplex.data.common.flows import GenerateTrainingDataForTesting
from jobflow import run_locally


def test_generate_training_data_for_testing(
        vasp_test_dir, test_dir, memory_jobstore, clean_dir
):

    path_to_struct = vasp_test_dir / "dft_ml_data_generation" / "POSCAR"
    potential_file_dir = test_dir / "fitting" / "ref_files" / "gap_file.xml"
    structure = Structure.from_file(path_to_struct)
    generate_data = GenerateTrainingDataForTesting().make(
        train_structure_list=[structure],
        cell_factor_sequence=[0.95, 1.0, 1.05],
        potential_filename=potential_file_dir,
        n_structures=1,
        steps=1,
    )

    responses = run_locally(
        generate_data, create_folders=True, ensure_success=False, store=memory_jobstore
    )  # atomate2 switched from pckl to json files for the trajectories --> job fails in its current state
