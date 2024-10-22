import os 
os.environ["OMP_NUM_THREADS"] = "1"

from autoplex.data.rss.jobs import do_rss
from jobflow import run_locally
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np


def test_gap_rss(test_dir, memory_jobstore, clean_dir):
    np.random.seed(42)
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index="0:5:1")
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    mlip_path = test_dir / "fitting/GAP"

    job = do_rss(mlip_type='GAP',
                iteration_index='0',
                mlip_path=mlip_path,
                structure=structures,
                scalar_pressure_method='exp',
                scalar_exp_pressure=100,
                scalar_pressure_exponential_width=0.2,
                scalar_pressure_low=0,
                scalar_pressure_high=50,
                max_steps=10,
                force_tol=0.1,
                stress_tol=0.1,
                Hookean_repul=False,
                write_traj=True,
                num_processes_rss=4,
                device="cpu",
                isol_es={14: -0.84696938})
    
    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    output = job.output.resolve(memory_jobstore)
    output_filter = []
    for i in output:
        if i is not None:
            output_filter.append(i)
   
    assert len(output_filter) == 2



# def test_jace_rss(test_dir, memory_jobstore):
#     np.random.seed(42)
#     test_files_dir = test_dir / "data/rss.extxyz"
#     atoms = read(test_files_dir, index="0:5:1")
#     structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
#     mlip_path = test_dir / "fitting/JACE"

#     job = do_rss(mlip_type='J-ACE',
#                 iteration_index='0',
#                 mlip_path=mlip_path,
#                 structure=structures,
#                 scalar_pressure_method='exp',
#                 scalar_exp_pressure=100,
#                 scalar_pressure_exponential_width=0.2,
#                 scalar_pressure_low=0,
#                 scalar_pressure_high=50,
#                 max_steps=10,
#                 force_tol=0.1,
#                 stress_tol=0.1,
#                 Hookean_repul=False,
#                 write_traj=True,
#                 num_processes_rss=4,
#                 device="cpu",
#                 isol_es={14: -0.84696938})
    
#     response = run_locally(
#         job,
#         create_folders=True,
#         ensure_success=True,
#         store=memory_jobstore
#     )

#     output = job.output.resolve(memory_jobstore)
#     output_filter = []
#     for i in output:
#         if i is not None:
#             output_filter.append(i)
   
#     assert len(output_filter) == 4

#     dir = Path('.')
#     path_to_job_files = list(dir.glob("job*"))
#     for path in path_to_job_files:
#         shutil.rmtree(path)


def test_nequip_rss(test_dir, memory_jobstore, clean_dir):
    np.random.seed(42)
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index="0:5:1")
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    mlip_path = test_dir / "fitting/NEQUIP"

    job = do_rss(mlip_type='NEQUIP',
                iteration_index='0',
                mlip_path=mlip_path,
                structure=structures,
                scalar_pressure_method='exp',
                scalar_exp_pressure=100,
                scalar_pressure_exponential_width=0.2,
                scalar_pressure_low=0,
                scalar_pressure_high=50,
                max_steps=10,
                force_tol=0.1,
                stress_tol=0.1,
                Hookean_repul=False,
                write_traj=True,
                num_processes_rss=4,
                device="cpu",
                isol_es={14: -0.84696938})
    
    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    output = job.output.resolve(memory_jobstore)
    output_filter = []
    for i in output:
        if i is not None:
            output_filter.append(i)
   
    assert len(output_filter) == 1


def test_m3gnet_rss(test_dir, memory_jobstore, clean_dir):
    np.random.seed(42)
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index="0:5:1")
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    mlip_path = test_dir / "fitting/M3GNET/m3gnet_results/training"

    job = do_rss(mlip_type='M3GNET',
                iteration_index='0',
                mlip_path=mlip_path,
                structure=structures,
                scalar_pressure_method='exp',
                scalar_exp_pressure=100,
                scalar_pressure_exponential_width=0.2,
                scalar_pressure_low=0,
                scalar_pressure_high=50,
                max_steps=10,
                force_tol=0.1,
                stress_tol=0.1,
                Hookean_repul=False,
                write_traj=True,
                num_processes_rss=4,
                device="cpu",
                isol_es={14: -0.84696938})
    
    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    output = job.output.resolve(memory_jobstore)
    output_filter = []
    for i in output:
        if i is not None:
            output_filter.append(i)
   
    assert len(output_filter) == 1



def test_mace_rss(test_dir, memory_jobstore, clean_dir):
    np.random.seed(42)
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index="0:5:1")
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    mlip_path = test_dir / "fitting/MACE"

    job = do_rss(mlip_type='MACE',
                iteration_index='0',
                mlip_path=mlip_path,
                structure=structures,
                scalar_pressure_method='exp',
                scalar_exp_pressure=100,
                scalar_pressure_exponential_width=0.2,
                scalar_pressure_low=0,
                scalar_pressure_high=50,
                max_steps=10,
                force_tol=0.1,
                stress_tol=0.1,
                Hookean_repul=False,
                write_traj=True,
                num_processes_rss=4,
                device="cpu",
                isol_es={14: -0.84696938})
    
    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    output = job.output.resolve(memory_jobstore)
    output_filter = []
    for i in output:
        if i is not None:
            output_filter.append(i)
   
    assert len(output_filter) == 1


        