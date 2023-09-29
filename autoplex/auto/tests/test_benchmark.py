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


class TestBenchmark(unittest.TestCase):
    def test_benchmark(self):
        assert # just noting ideas for now
        assert right sturcture has right mpid and right rms


if __name__ == '__main__':
    unittest.main()
