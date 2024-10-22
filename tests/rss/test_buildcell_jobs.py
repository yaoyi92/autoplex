from __future__ import annotations
from autoplex.data.rss.jobs import RandomizedStructure
import os


def test_extract_elements():
    rs = RandomizedStructure()
    elements = rs._extract_elements("SiO2")
    assert elements == {"Si": 1, "O": 2}
    
    elements = rs._extract_elements("H2O")
    assert elements == {"H": 2, "O": 1}
    
    elements = rs._extract_elements("C6H12O6")
    assert elements == {"C": 6, "H": 12, "O": 6}


def test_make_species():
    rs = RandomizedStructure()
    elements = {"Si": 1, "O": 2}
    species = rs._make_species(elements)
    assert species == "Si%NUM=1,O%NUM=2"
    
    elements = {"H": 2, "O": 1}
    species = rs._make_species(elements)
    assert species == "H%NUM=2,O%NUM=1"


def test_is_metal():
    rs = RandomizedStructure()
    assert rs._is_metal("Fe") == True
    assert rs._is_metal("Si") == False


def test_make_minsep():
    rs = RandomizedStructure()
    radii = {"Si": 1.1, "O": 0.66}
    minsep = rs._make_minsep(radii)
    assert "Si-Si=2.2" in minsep  # r1 * 2.0
    assert "Si-O=1.32" in minsep  # (r1 + r2) / 2 * 1.5
    assert "O-O=1.32" in minsep   # r1 * 2.0


def test_update_buildcell_options():
    rs = RandomizedStructure()
    options = {'VARVOL': 20, 'SPECIES': 'Si%NUM=1,O%NUM=2'}
    buildcell_parameters = ['VARVOL=15',
                            'NFORM=1-7',
                            ]
    buildcell_update = rs._update_buildcell_options(options, buildcell_parameters)
    print("Updated buildcell parameters:", buildcell_update)
    assert 'VARVOL=20' in buildcell_update
    assert 'SPECIES=Si%NUM=1,O%NUM=2' in buildcell_update


def test_output_from_scratch(memory_jobstore):
    from jobflow import run_locally
    from ase.io import read
    from pathlib import Path
    import shutil
    job = RandomizedStructure(struct_number=3,
                              tag='SiO2',
                              output_file_name='random_structs.extxyz',
                              buildcell_options={'VARVOL': 20,
                                                 'SYMMOPS':'1-2'},
                              num_processes=2).make()
    
    responses = run_locally(job, ensure_success=True, create_folders=True, store=memory_jobstore)
    assert Path(job.output.resolve(memory_jobstore)).exists()
    atoms = read(Path(job.output.resolve(memory_jobstore)), index=":")
    assert len(atoms) == 3

    dir = Path('.')
    path_to_job_files = list(dir.glob("job*"))
    for path in path_to_job_files:
        shutil.rmtree(path)

def test_output_from_cell_seed(test_dir, memory_jobstore):
    from jobflow import run_locally
    from ase.io import read
    from pathlib import Path
    import shutil
    test_files_dir = test_dir / "data/SiO2.cell"
    job = RandomizedStructure(struct_number=3,
                              cell_seed_path=test_files_dir,
                              num_processes=2).make()
    
    responses = run_locally(job, ensure_success=True, create_folders=True, store=memory_jobstore)
    assert Path(job.output.resolve(memory_jobstore)).exists()
    atoms = read(Path(job.output.resolve(memory_jobstore)), index=":")
    assert len(atoms) == 3
    
    dir = Path('.')
    path_to_job_files = list(dir.glob("job*"))
    for path in path_to_job_files:
        shutil.rmtree(path)



    