from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx

from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Vasprun
from autoplex.data.flows import DataGenerator, IsoAtomMaker


class TestDataGen(unittest.TestCase):
    def test_datageneration(self):
        test_structure = Structure(
            lattice=[
                [3.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 3.0],],
            species=["Mo", "C", "K"],
            coords=[[0.25, 0.25, 0.25], [0, 0, 0], [0, 0, 0]],)
        test_mpid = "mp-test"
        test_species = test_structure.species()

        flow = DataGenerator().make(structure_mock, mpid_mock, supercell_matrix_mock)
        flow = IsoAtomMaker().make(species_mock) # in Janine's phonon implementation gucken

        assert len(flow.jobs) == expected_number_of_jobs #???
        assert isolated_atoms_no_duplicate_atoms # just noting ideas for now
        assert sc true/false



if __name__ == '__main__':
    unittest.main()
