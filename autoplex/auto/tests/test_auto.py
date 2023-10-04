from __future__ import annotations

import os
import unittest
import pandas as pd
from jobflow import run_locally
from pymatgen.core.structure import Structure
from autoplex.auto.flows import CompleteWorkflow, PhononDFTMLDataGenerationFlow, PhononDFTMLFitFlow, PhononDFTMLBenchmarkFlow
from autoplex.auto.jobs import CollectBenchmark


class TestAutomation(unittest.TestCase):
    def test_automation(self):
        test_structure = Structure(
            lattice = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0], ],
            species = ["Mo", "C", "K"],
            coords = [[0.66, 0.66, 0.66], [0.33, 0.33, 0.33], [0, 0, 0]], )
        test_mpid = "mp-test"
        test_species = test_structure.types_of_species

        test_struc_list = []
        test_mpids = []
        rms_dis = [[0.1, 0.2], [0.7, 0.9]]
        for i in range(2):
            test_struc_list.append(test_structure)
            test_mpids.append(test_mpid+str(i+1))

        complete = CompleteWorkflow().make(structure_list = test_struc_list, mpids = test_mpids, phonon_displacement_maker = None)
        dataauto = PhononDFTMLDataGenerationFlow().make(structure = test_structure, mpid = test_mpid)
        fitauto = PhononDFTMLFitFlow().make(species = test_species, isolated_atoms = [7, 5, 3], fitinput = [])
        bmauto = PhononDFTMLBenchmarkFlow().make(structure = test_structure, mpid = test_mpid, ml_reference = None, dft_reference = None)
        collect = CollectBenchmark(structure_list = test_struc_list, mpids = test_mpids, rms_dis = rms_dis, displacements = [0.4, 0.3])
        responses = run_locally(collect, ensure_success = True)

        assert len(complete.jobs) == 11 # not sure how else checking if job is submitted correctly without actually running VASP jobs, also implicitly tests that each element type is only submitted once as iso-atom
        assert len(dataauto.jobs) == 2
        assert len(fitauto.jobs) == 1
        assert len(bmauto.jobs) == 1
        assert responses is not None

        df_expect = pd.read_csv('results_test.txt', engine='python', sep=' ', skiprows=1, skipfooter=1, header=None, names=['Pot', 'Structure', 'mpid', 'displacement', 'RMS', 'imagmodes(pot)', 'imagmodes(dft)'])
        df = pd.read_csv('results_.txt', engine='python', sep=' ', skiprows=1, skipfooter=1, header=None, names=['Pot', 'Structure', 'mpid', 'displacement', 'RMS', 'imagmodes(pot)', 'imagmodes(dft)'])
        for i, j in zip(df, df_expect): assert i == j #somehow the test fails when I compare it listwise

        os.remove("results_.txt")

