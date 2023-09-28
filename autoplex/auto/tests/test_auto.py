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


class TestAutomation(unittest.TestCase):
    def test_automation(self):
        assert complete wf  # for now just noting ideas
        assert individual steps


if __name__ == '__main__':
    unittest.main()
