from pymatgen.core.structure import Structure
from autoplex.data.common.jobs import generate_randomized_structures
from autoplex.data.common.utils import (
    mc_rattle,
    random_vary_angle,
    scale_cell,
    std_rattle
)
import numpy as np


def test_generate_randomized_structures():
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
                        volume_scale_factor_range=[0.90,1.10], 
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

def test_mc_rattle():
    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    mc_rattle_job=mc_rattle(structure=structure, n_structures=10)

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

    std_rattle_job=std_rattle(structure=structure, n_structures=10)

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

    random_vary_angle_job=random_vary_angle(structure=structure, n_structures=10)

    # check if correct number of structures are generated
    assert len(random_vary_angle_job) == 10
    for struct in random_vary_angle_job:
        # check if all outputs are Structure objects
        assert isinstance(struct, Structure)
        # check if the distorted structures have the same number of sites as the original structure
        assert struct.num_sites == structure.num_sites
        # check lattice parameters are reasonably close to those before distorting
        assert np.allclose((struct.lattice.matrix).all(), (structure.lattice.matrix).all(), atol= 0.5)


# adapt to check for each input possible e.g. inputting range/manual scale_factors?
def test_scale_cell():

    structure = Structure(
        lattice=[[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]],
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )

    scale_cell_job=scale_cell(structure=structure, volume_scale_factor_range=[0.90, 1.10], n_structures=10)

    # check if correct number of structures are generated
    assert len(scale_cell_job) == 10
    for struct in scale_cell_job:
        # check if all outputs are Structure objects
        assert isinstance(struct, Structure)
        # check if the distorted structures have the same number of sites as the original structure
        assert struct.num_sites == structure.num_sites
        # check lattice parameters are within +-10% of original value
        assert np.allclose(np.abs(np.array(struct.lattice.abc) - np.array(structure.lattice.abc)), 0, atol=0.1 * np.array(structure.lattice.abc))
