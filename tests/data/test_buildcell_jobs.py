from __future__ import annotations
from autoplex.data.rss.jobs import RandomizedStructure
import os

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1

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
    assert "Si-Si=2.2" in minsep
    assert "Si-O=1.32" in minsep
    assert "O-O=1.32" in minsep

def test_update_buildcell_options():
    rs = RandomizedStructure()
    updates = {'VARVOL': 20, 'SPECIES': 'Si%NUM=1,O%NUM=2'}
    rs._update_buildcell_options(updates)
    assert 'VARVOL=20' in rs.buildcell_options
    assert 'SPECIES=Si%NUM=1,O%NUM=2' in rs.buildcell_options