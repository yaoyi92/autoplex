(flows)=

# Out of the box workflow

This tutorial will demonstrate how to use `autoplex` with its default setup and settings.

> ℹ️ The default setting might not be sufficient or not suitable in any other way for your calculations. Carefully check your results with the default setup and adjust the settings when needed.

## General workflow

The complete workflow of `autoplex` involves the data generation (including the execution of VASP calculations), the fitting of the  machine-learned interatomic potential (MLIP) and the benchmark to the DFT results.

Let us start by importing all the necessary modules:

```python
from jobflow.core.flow import Flow
from mp_api.client import MPRester
from autoplex.auto.phonons.flows import CompleteDFTvsMLBenchmarkWorkflow
```
We will use [jobflow](https://github.com/materialsproject/jobflow) to control the execution of our jobs in form of flows and jobs.
Using the `MPRester` is a convenient way to draw structures from the Materials Project database using their MP-ID.
The only module we need to import from `autoplex` is the `CompleteDFTvsMLBenchmarkWorkflow`.


Next we are going to construct the workflow based on dia-Si ([*mp-149*](https://next-gen.materialsproject.org/materials/mp-149?material_ids=mp-149)) and another hexagonal allotrope ([*mp-165*](https://next-gen.materialsproject.org/materials/mp-165?material_ids=mp-165)). 

```python
mpr = MPRester(api_key='YOUR_MP_API_KEY')
structure_list = []
benchmark_structure_list = []
mpids = ["mp-149", "mp-165"]
mpbenchmark = ["mp-149", "mp-165"]
for mpid in mpids:
    structure = mpr.get_structure_by_material_id(mpid)
    structure_list.append(structure)
for mpbm in mpbenchmark:
    bm_structure = mpr.get_structure_by_material_id(mpbm)
    benchmark_structure_list.append(bm_structure)

complete_flow = CompleteDFTvsMLBenchmarkWorkflow().make(
    structure_list=struc_list, mp_ids=mpids, 
    benchmark_structures=benchmark_structure_list, benchmark_mp_ids=mpbenchmark)

autoplex_flow = Flow([complete_flow], name="tutorial", output=None, uuid=None, hosts=None)
```
The only information we need to provide is which structures we want to calculate and use for the MLIP fitting and which structures we want to benchmark to.
