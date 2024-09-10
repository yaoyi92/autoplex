from mp_api.client import MPRester
from autoplex.auto.phonons.flows import CompleteDFTvsMLBenchmarkWorkflow, SettingsTestMaker
from jobflow_remote import submit_flow, set_run_config
from atomate2.vasp.powerups import update_user_incar_settings
from jobflow import run_locally
from atomate2.settings import Atomate2Settings
from atomate2.forcefields.jobs import CHGNetStaticMaker
#Atomate2Settings.VASP_INCAR_UPDATES={"NPAR": 4}

mpr = MPRester(api_key='Z4aKTAgeEudmS0bMPkKVS3EtOnej1zah')
structure_list = []
benchmark_structure_list = []
mpids = ["mp-117"]
mpbenchmark =["mp-117"]
for mpid in mpids:
    structure = mpr.get_structure_by_material_id(mpid)
    structure_list.append(structure)
for mpbm in mpbenchmark:
    bm_structure = mpr.get_structure_by_material_id(mpbm)
    benchmark_structure_list.append(bm_structure)

autoplex_flow = SettingsTestMaker(adaptive_supercell_settings={"min_length":10, "min_atoms":10}, DFT_Maker=CHGNetStaticMaker()).make(
    structure_list=structure_list)


run_locally(autoplex_flow)