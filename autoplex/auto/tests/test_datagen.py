from __future__ import annotations

import unittest
import inspect
from pymatgen.core.structure import Structure
from autoplex.data.flows import DataGenerator, IsoAtomMaker


class TestDataGen(unittest.TestCase):
    def test_datageneration(self):
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
        test_species = test_structure.types_of_species

        data = DataGenerator().make(structure=test_structure, mpid=test_mpid)
        isoatom = []
        for species in test_species:
            isoatom.append(IsoAtomMaker().make(species))

        assert (
            len(data.jobs) == 2
        )  # not sure how else checking if job is submitted correctly without actually running VASP jobs
        assert len((isoatom)) == 3
        assert [
            items.default
            for items in inspect.signature(DataGenerator).parameters.values()
            if items.name == "sc"
        ][
            0
        ] is False  # important to avoid accidentally wayyy too large workflows
