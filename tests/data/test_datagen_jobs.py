from pymatgen.core.structure import Structure
from autoplex.data.common.jobs import generate_randomized_structures, convert_to_extxyz, plot_force_distribution
from autoplex.data.common.utils import (
    mc_rattle,
    random_vary_angle,
    scale_cell,
    std_rattle
)
import numpy as np


# test distort_type=0, i.e. volume distortion
def test_generate_randomized_structures_distort_type_0():
    from jobflow import run_locally

    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    rand_structs_job = generate_randomized_structures(
        structure=structure,
        distort_type=0,
        n_structures=10,
        volume_scale_factor_range=[0.90, 1.10],
        rattle_type=0)

    responses = run_locally(rand_structs_job, create_folders=False, ensure_success=True)

    # check if correct number of valid structure objects are generated
    for uuid, response_collection in responses.items():
        for k, response in response_collection.items():
            # check if correct number of structures are generated
            assert 11 == len(response.output)
            for struct in response.output:
                # check if all outputs are Structure objects
                assert isinstance(struct, Structure)


# test distort_type=1, i.e. angle distortion
def test_generate_randomized_structures_distort_type_1():
    from jobflow import run_locally

    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    rand_structs_job = generate_randomized_structures(
        structure=structure,
        distort_type=1,
        n_structures=10,
        min_distance=1.5,
        angle_percentage_scale=10,
        angle_max_attempts=1000,
        rattle_type=0)

    responses = run_locally(rand_structs_job, create_folders=False, ensure_success=True)

    # check if correct number of valid structure objects are generated
    for uuid, response_collection in responses.items():
        for k, response in response_collection.items():
            # check if correct number of structures are generated
            assert 10 == len(response.output)
            for struct in response.output:
                # check if all outputs are Structure objects
                assert isinstance(struct, Structure)


# test distort_type=2, i.e. simultaneous volume and angle distortion
def test_generate_randomized_structures_distort_type_2():
    from jobflow import run_locally

    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    rand_structs_job = generate_randomized_structures(
        structure=structure,
        distort_type=2,
        n_structures=10,
        volume_scale_factor_range=[0.90, 1.10],
        min_distance=1.5,
        angle_percentage_scale=10,
        angle_max_attempts=1000,
        rattle_type=0)

    responses = run_locally(rand_structs_job, create_folders=False, ensure_success=True)

    # check if correct number of valid structure objects are generated
    for uuid, response_collection in responses.items():
        for k, response in response_collection.items():
            # check if correct number of structures are generated
            assert 11 == len(response.output)
            for struct in response.output:
                # check if all outputs are Structure objects
                assert isinstance(struct, Structure)


def test_mc_rattle():
    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    mc_rattle_job = mc_rattle(structure=structure, n_structures=10)

    # check if correct number of structures are generated
    assert len(mc_rattle_job) == 10
    for struct in mc_rattle_job:
        # check if all outputs are Structure objects
        assert isinstance(struct, Structure)
        # check if the rattled structures have the same number of sites as the original structure
        assert struct.num_sites == structure.num_sites
        # check if lattice parameters are unchanged
        assert (struct.lattice.matrix).all() == (structure.lattice.matrix).all()
        # check if atom positions are reasonably close to positions before rattling
        assert np.allclose(struct.frac_coords, structure.frac_coords, atol=0.05)


def test_std_rattle():
    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    std_rattle_job = std_rattle(structure=structure, n_structures=10)

    # check if correct number of structures are generated
    assert len(std_rattle_job) == 10
    for struct in std_rattle_job:
        # check if all outputs are Structure objects
        assert isinstance(struct, Structure)
        # check if the rattled structures have the same number of sites as the original structure
        assert struct.num_sites == structure.num_sites
        # check if lattice parameters are unchanged
        assert (struct.lattice.matrix).all() == (structure.lattice.matrix).all()
        # check if atom positions are reasonably close to positions before rattling
        assert np.allclose(struct.frac_coords, structure.frac_coords, atol=0.05)


def test_random_vary_angle():
    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    random_vary_angle_job = random_vary_angle(structure=structure, n_structures=10)

    # check if correct number of structures are generated
    assert len(random_vary_angle_job) == 10
    for struct in random_vary_angle_job:
        # check if all outputs are Structure objects
        assert isinstance(struct, Structure)
        # check if the distorted structures have the same number of sites as the original structure
        assert struct.num_sites == structure.num_sites
        # check lattice parameters are reasonably close to those before distorting
        assert np.allclose((struct.lattice.matrix).all(), (structure.lattice.matrix).all(), atol=0.5)


# adapt to check for each input possible e.g. inputting range/manual scale_factors?
def test_scale_cell():
    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    scale_cell_job = scale_cell(structure=structure, volume_scale_factor_range=[0.90, 1.10], n_structures=10)

    # check if correct number of structures are generated
    assert len(scale_cell_job) == 11
    for struct in scale_cell_job:
        # check if all outputs are Structure objects
        assert isinstance(struct, Structure)
        # check if the distorted structures have the same number of sites as the original structure
        assert struct.num_sites == structure.num_sites
        # check lattice parameters are within +-10% of original value
        assert np.allclose(np.abs(np.array(struct.lattice.abc) - np.array(structure.lattice.abc)), 0,
                           atol=0.1 * np.array(structure.lattice.abc))


def test_convert_to_extxyz(test_dir, memory_jobstore, clean_dir):
    from jobflow import run_locally
    class Job:  # create a dummy job object
        def __init__(self, dir_name):
            self.output = self.Output(dir_name)

        class Output:
            def __init__(self, dir_name):
                self.dir_name = dir_name

            def as_dict(self):
                return {"dir_name": self.dir_name}

    dir = test_dir / "pkls"
    file = "LiCl_10.pkl"
    job = Job(dir)

    conv = convert_to_extxyz(job.output, file, "bulk", "10")

    responses = run_locally(
        conv, create_folders=True, ensure_success=False, store=memory_jobstore
        # atomate2 swithced from pkl to json files for the trajectories, therfore success False. This job is also not needed for the main workflow
    )


def test_plot_force_distribution(test_dir, memory_jobstore, clean_dir):
    from jobflow import run_locally
    import glob
    import os

    dir = test_dir / "fitting"

    plot = plot_force_distribution(1.0, str(dir))

    responses = run_locally(
        plot, create_folders=False, ensure_success=False, store=memory_jobstore
    )

    assert os.path.isfile("total_data.png")

    files_remove = glob.glob("*.png")

    for file in files_remove:
        try:
            os.remove(file)
            print(f'Removed file: {file}')
        except Exception as e:
            print(f'Error removing file {file}: {e}')



