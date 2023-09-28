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


class TestDataGen(unittest.TestCase):
    def test_datageneration(self):
        assert isolated_atoms_no_duplicate_atoms # just noting ideas for now
        assert sc true/false


if __name__ == '__main__':
    unittest.main()
