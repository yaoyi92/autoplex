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


class TestFitting(unittest.TestCase):
    def test_fitting(self):
        assert hyperparametrs set correctly # just noting ideas for now
        assert gap git successful
        assert 2body (maybe 3body) + soap set


if __name__ == '__main__':
    unittest.main()
