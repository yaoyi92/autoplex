(benchmark)=

*Tutorials written by Christina Ertural ([christina.ertural@bam.de](mailto:christina.ertural@bam.de)).*

# Benchmark

This tutorial will help you understand all the `autoplex` benchmark specifications.

## General settings

For the benchmark, you do not have to worry about a lot of settings. The crucial part here is the number of benchmark structures you are interested in.

```python
from mp_api.client import MPRester
from autoplex.auto.phonons.flows import CompleteDFTvsMLBenchmarkWorkflow
from atomate2.common.schemas.phonons import PhononBSDOSDoc
from monty.serialization import loadfn

mpr = MPRester(api_key='YOUR_MP_API_KEY')
structure_list = []
benchmark_structure_list = []
mpids = ["mp-22905"]
mpbenchmark = ["mp-22905"]
dft_data = loadfn("/path/to/DFT/ref/data/PhononBSDOSDoc_mp_22905.json")
dft_reference: PhononBSDOSDoc = dft_data["output"]
for mpid in mpids:
    structure = mpr.get_structure_by_material_id(mpid)
    structure_list.append(structure)
for mpbm in mpbenchmark:
    bm_structure = mpr.get_structure_by_material_id(mpbm)
    benchmark_structure_list.append(bm_structure)

complete_flow = CompleteDFTvsMLBenchmarkWorkflow().make(
    structure_list=structure_list, mp_ids=mpids, 
    benchmark_structures=benchmark_structure_list, benchmark_mp_ids=mpbenchmark, 
    dft_references=[dft_reference])
```
In case you have pre-existing DFT calculations, you can pass them as a list via the `dft_references` parameters. Make sure, it is the same order as in the benchmark MP-IDs and structures.
It is important to provide the pre-existing DFT data in form of a `PhononBSDOSDoc` task document object (from `atomate2`). Without any DFT reference calculations given, `autoplex` will automatically execute the VASP calculations. A mix of pre-existing and missing DFT references is not supported.

## Error metrics
`autoplex` automatically provides you with a phonon bandstructure comparison plot, a q-point wise RMSE plot and an overall RMSE value (`results_XY.txt` file). For examples see [here](../flows/flows.md#output-and-results).

## Run a benchmark with a pre-existing DFT calculation and GAP potential

If you want to run or repeat the benchmark with your pre-existing DFT calculation and your pre-existing GAP potential, you can use the following Python script as a template.
It is important to provide the pre-existing DFT data in form of a `PhononBSDOSDoc` task document object (from `atomate2`).

```python
#!/usr/bin/env python

import os
from atomate2.common.schemas.phonons import PhononBSDOSDoc
from atomate2.forcefields.jobs import GAPRelaxMaker, GAPStaticMaker
from mp_api.client import MPRester
from autoplex.benchmark.phonons.flows import PhononBenchmarkMaker
from atomate2.forcefields.flows.phonons import PhononMaker
from autoplex.benchmark.phonons.jobs import write_benchmark_metrics
from monty.serialization import loadfn
from jobflow import SETTINGS
from jobflow import run_locally

os.environ["OMP_NUM_THREADS"] = "48"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

store = SETTINGS.JOB_STORE
# connect to the job store
store.connect()

mpr = MPRester(api_key = 'YOUR_API_KEY')
mpid = "mp-22905"
structure =  mpr.get_structure_by_material_id(mpid)
dft_data = loadfn("/path/to/DFT/ref/data/PhononBSDOSDoc_mp_22905.json")
dft_reference: PhononBSDOSDoc = dft_data["output"]
potential_filename = "/path/to/GAP/file/gap_file.xml"

phojob = PhononMaker(
        bulk_relax_maker=GAPRelaxMaker(calculator_kwargs={"args_str": "IP GAP", "param_filename": potential_filename}, 
        relax_cell=True, relax_kwargs={"interval": 500, "fmax": 0.00001}, steps=10000),
        phonon_displacement_maker=GAPStaticMaker(calculator_kwargs={"args_str": "IP GAP", "param_filename": potential_filename}),
        static_energy_maker=GAPStaticMaker(calculator_kwargs={"args_str": "IP GAP", "param_filename": potential_filename}),
        store_force_constants=False, min_length=18,
        generate_frequencies_eigenvectors_kwargs={"units": "THz"}).make(structure=structure)
        
bm = PhononBenchmarkMaker(name="Benchmark").make(
    structure=structure, benchmark_mp_id = "mp-22905", 
    ml_phonon_task_doc = phojob.output, dft_phonon_task_doc = dft_reference)

comp_bm = write_benchmark_metrics(
            ml_models=["GAP"],
            benchmark_structures=[structure],
            benchmark_mp_ids=["mp-22905"],
            metrics=bm.output,
            displacements=[0.01],
        )

run_locally([phojob, bm, comp_bm], create_folders=True, store=store)
```
If you use another [`ForceFieldRelaxMaker` and `ForceFieldStaticMaker`](https://github.com/materialsproject/atomate2/blob/main/src/atomate2/forcefields/jobs.py), you can switch from GAP to one of the other [MLIP potentials](../fitting/fitting.md#fitting-potentials).

