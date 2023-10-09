#!/usr/bin/env python

from fireworks import LaunchPad
from jobflow.managers.fireworks import flow_to_workflow
from jobflow.core.flow import Flow
from autoplex.auto.jobs import PhononMLCalculationJob
from pymatgen.core import Structure

# tutorial for calc with existing potential

structure = Structure.from_file("POSCAR")  # or e.g. cif or xyz format
name = "NameOfYourWorkflow"

existing_pot = PhononMLCalculationJob(
    structure=structure,
    displacements=[0.1],
    min_length=20,
    ml_dir="/path/to/your/GAP/pot/gap.xml",
)

tutorial_flow = Flow(
    [existing_pot], output=None, name=name + "-AutoPLEX-Flow", uuid=None, hosts=None
)
wf = flow_to_workflow(tutorial_flow)

# submit the workflow to the FireWorks launchpad
lpad = LaunchPad.auto_load()
lpad.add_wf(wf)
