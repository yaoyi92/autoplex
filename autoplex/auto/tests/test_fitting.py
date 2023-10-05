from __future__ import annotations

import os
import unittest
from pathlib import Path
from jobflow import run_locally
from autoplex.fitting.flows import MLIPFitMaker

CurrentDir = Path(__file__).absolute().parent

class TestFitting(unittest.TestCase):
    def test_fitting(self):
        gapfit = MLIPFitMaker().make(species_list = ["Li", "Cl"], iso_atom_energy = [5, 3], fitinput = [{'dir':[f"{CurrentDir}/files"]}])
        responses = run_locally(gapfit, ensure_success = True)

        assert responses is not None
        #assert hyperparametrs set correctly/check if 2body (maybe 3body) + soap set
        #fitkwargs e0 update

        for file in os.listdir("."):
            if os.path.isfile(file) and file.startswith("gap.xml") \
                    or file.startswith("trainGAP") or file.startswith("std_"): os.remove(file)


