# from __future__ import annotations
#
# import os
# import unittest
# from pathlib import Path
# from jobflow import run_locally
# from autoplex.fitting.flows import MLIPFitMaker
#
# def test_fitting_flow(test_dir):
#
#     outcar = test_dir / "fitting"
#     #print(outcar)
#     #print(test_dir.parent)
#     gapfit = MLIPFitMaker().make(
#         species_list=["Li", "Cl"],
#         iso_atom_energy=[5, 3],
#         fit_input=[{"dir": [outcar]}],
#     )
#     responses = run_locally(gapfit, ensure_success=True)
#
#         # assert responses is not None
#         # # assert hyperparametrs set correctly/check if 2body (maybe 3body) + soap set
#         # # fitkwargs e0 update
#         #
#         # for file in os.listdir("."):
#         #     if (
#         #         os.path.isfile(file)
#         #         and file.startswith("gap.xml")
#         #         or file.startswith("trainGAP")
#         #         or file.startswith("std_")
#         #     ):
#         #         os.remove(file)
#         #pass
