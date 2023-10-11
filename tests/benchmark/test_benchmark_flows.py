from __future__ import annotations

import unittest

from pymatgen.core.structure import Structure
from autoplex.benchmark.flows import PhononBenchmarkMaker


class TestBenchmark(unittest.TestCase):
    def test_benchmark(self):
        test_structure = Structure(
            lattice=[
                [3.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 3.0],
            ],
            species=["Mo", "C", "K"],
            coords=[[0.66, 0.66, 0.66], [0.33, 0.33, 0.33], [0, 0, 0]],
        )
        test_mpid = "mp-test"
        benchmark = PhononBenchmarkMaker().make(
            structure=test_structure,
            mp_id=test_mpid,
            ml_phonon_task_doc=None,
            dft_phonon_task_doc=None,
        )

        assert len(benchmark.jobs) == 1
        # assert bm ml/dft is working ==> need to construct a test_ml_reference and test_dft_reference
