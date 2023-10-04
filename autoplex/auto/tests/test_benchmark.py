from __future__ import annotations

import unittest

import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytest import approx

from pymatgen.core.structure import Structure
from autoplex.benchmark.flows import PhononBenchmarkMaker


class TestBenchmark(unittest.TestCase):
    def test_benchmark(self):
        test_structure = Structure(
            lattice = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0], ],
            species = ["Mo", "C", "K"],
            coords = [[0.66, 0.66, 0.66], [0.33, 0.33, 0.33], [0, 0, 0]], )
        test_mpid = "mp-test"
        benchmark = PhononBenchmarkMaker().make(structure = test_structure, mpid = test_mpid, ml_reference = None, dft_reference = None)

        assert len(benchmark.jobs) == 1
        #assert bm ml/dft is working ==> need to construct a test_ml_reference and test_dft_reference

