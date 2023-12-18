from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.sets.core import StaticSetGenerator
from autoplex.auto.flows import CompleteDFTvsMLBenchmarkWorkflow
from fireworks import LaunchPad
from jobflow.core.flow import Flow
from jobflow.managers.fireworks import flow_to_workflow
from jobflow.utils.graph import to_mermaid
from mp_api.client import MPRester

# Please be aware that you need to use your new API key here.
mpr = MPRester(api_key="your MP API key")
# generate the structure list by using Materials Project IDs
struc_list = []
mpids = ["mp-149", "mp-165"]
for mpid in mpids:
    struc = mpr.get_structure_by_material_id(mpid)
    struc_list.append(struc)

# accuracy setting (grid_density, n_struc, symprec) are very low
phonon_stat = BaseVaspMaker(
    input_set_generator=StaticSetGenerator(
        user_kpoints_settings={"grid_density": 1},
    )
)  # reduced the accuracy for test calculations
complete_flow = CompleteDFTvsMLBenchmarkWorkflow(
    n_struc=1, displacements=[0.01], symprec=1e-4, uc=False
).make(structure_list=struc_list, mp_ids=mpids, phonon_displacement_maker=phonon_stat)

autoplex_flow = Flow(
    [complete_flow], output=None, name="Si-AutoPLEX-Flow", uuid=None, hosts=None
)

graph_source = to_mermaid(autoplex_flow, show_flow_boxes=True)
print(graph_source)  # print text to generate a pretty mermaid graph (mermaid.live)

wf = flow_to_workflow(autoplex_flow)

# submit the workflow to the FireWorks launchpad
lpad = LaunchPad.auto_load()
lpad.add_wf(wf)
