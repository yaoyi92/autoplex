*Tutorial written by Aakash Naik (aakash.naik@bam.de).*

# Jobflow-remote setup

This will result in a setup for automation where 
1. We will add/submit job to db on your local machine.
2. Jobs will be executed on your remote custer.

# Installation

## on your local machine
1. Create a new env > `conda create -n autoplex python=3.10`. (You can choose any other env name.)
2. Activate your env > `conda activate autoplex` 
3. Clone the jobflow remote repository using `git clone https://github.com/Matgenix/jobflow-remote.git`
4. Switch to interactive branch (use `git checkout remotes/origin/interactive`) and install it via `pip install .` in your env.
5. Install autoplex > In your local autoplex directory: `pip install -e .[strict]`. 
6. Activate your env and run `jf project generate --full YOUR_PROJECT_NAME`. 
This will generate an empty project config file in your home directory. 
You can find this file inside `~/.jfremote` 
(This is optional, a config file is provided here: [test_project.yaml](test_project.yaml), 
you can simply copy this file to the  `~/.jfremote` directory. You will need to create `~/.jfremote` directory in your home.)


## on your remote cluster
7. Repeat step 1,2,3,4 and 5 on your remote cluster.
8. Now setup atomate2 config as usual.
Just `atomate2/config/atomate2.yaml`. (We do not need to set up jobflow.yaml in atomate2/config)

Below is an example `atomat2.yaml` config file
```yaml
VASP_CMD:  your hpc vasp_std cmd
VASP_GAMMA_CMD:  your hpc vasp_gam cmd
LOBSTER_CMD: your hpc lobster cmd
```

9. Add environment variable to your ~/.bashrc `export ATOMATE2_CONFIG_FILE="/path/to/atomate2/config/atomate2.yaml"`

## Changes to be done in the config file - on your local machine
1. Set paths to base, tmp, log, daemon dir. Best would be, simply creating empty dirs in your `~/.jfremote` directory. 
Use the paths as provided in sample config file for reference.
2. Under the `workers` section of the yaml, change worker name from `example_worker` to your liking, set `work_dir` 
(directory where calcs will be run), set `pre_run` command (use to activate the environment before job execution), 
set `user` (this your username on your remote cluster)  
3. In `queue` section, just change details as per your mongodb (admin username password, host, port, name)


# Check if your setup works correctly

> Note: If you have any password protected key in your `~/.ssh` directory worker might fail to start. To overcome this, temporarily move your passphrase procted keys from `~/.ssh` directory to some other directory before starting the runner.

1. `jf project check -w example_worker` 
(If everything is setup correctly, you will get asked for password and OTP and will exit with a green tick in few secs.)
2. `jf project check --errors` this will check all even your MongoDB connection is proper or not. 
If anything fails, please check the config file.


# Getting started

1. Run `jf admin reset` (Do not worry, this will reset your db, necessary to do only once. 
You can skip this if you want to keep the data in your db.)
2. `jf runner start -s -i` 

You will be prompted with a question "Do you want to open the connection for the host of the XXX worker?" 
Answer "y". And then you should be prompted for password and OTP.
After that you can quit the interactive mode with ctrl+c. The runner should now be working fine until the connection drops.
 
During the starting of the runner, you will probably see a few error/warnings. 
First, a warning that the password may be echoed. Ignore it, it should not.  

3. `jf runner status` (this should return status of runner as `running`, if everything is set up correctly)


# Example job scripts to test (Add/Submit jobs to DB from your local machine)

## Simple python job

```python
from jobflow_remote.utils.examples import add
from jobflow_remote import submit_flow
from jobflow import Flow

job1 = add(1, 2)
job2 = add(job1.output, 2)

flow = Flow([job1, job2])

resources = {"nodes": N, "partition": "name", "time": "01:00:00", "ntasks": ntasks, "qverbatim": "#SBATCH --get-user-env",
             "mail_user": "your_email@adress", "mail_type": "ALL"}

print(submit_flow(flow, worker="example_worker", resources=resources, project="test_project")) 
# Do not forget to change worker and project name to what you se tup in the jobflow remote config file.
```

## VASP relax job using atomate2 workflow

```python
from jobflow_remote.utils.examples import add
from jobflow_remote import submit_flow
from jobflow import Flow
from mp_api.client import MPRester
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.powerups import update_user_incar_settings 


mpid = "mp-22862"
mr = MPRester(api_key='YOUR_MP_API_KEY')
struct = mr.get_structure_by_material_id(mpid)

# we use the same structure (mp-22862) here and instantiate the workflow
relax_job = DoubleRelaxMaker().make(structure=struct)

relax_job = update_user_incar_settings(relax_job, {"NPAR": 4})

# You can also pass exe_config for the worker using exe_config in submit flow. Below is an example 
# exec_config={"pre_run": "source activate autoplex \n module load slurm_setup \n module load vasp/6.1.2"}

resources = {"nodes": N, "partition": "name", "time": "01:00:00", "ntasks": ntasks, "qverbatim": "#SBATCH --get-user-env",
             "mail_user": "your_email@adress", "mail_type": "ALL"}

print(submit_flow(relax_job, worker="example_worker", resources=resources, project="test_project"))
```
It is crucial to set `"qverbatim": "#SBATCH --get-user-env"` to make sure the same environment is used on your remote cluster.

# Setting different workers for different job types

This is very much similar to how we do in atomate2, jobflow-remote provides a specific utility for this.
```python
from jobflow_remote import set_run_config
```
An example use case can be found [here](https://matgenix.github.io/jobflow-remote/user/tuning.html#jobconfig)

# Querying completed jobs from DB using jobflow-remote Python API

```python
from jobflow_remote import get_jobstore

js = get_jobstore(project_name='YOUR_PROJECT_NAME') 
js.connect()
result = js.query(criteria={"name": "generate_frequencies_eigenvectors"},load=True) 
# example query for completed phonon workflow runs
# the query methods are the same as in atomate2 basically, 
for i in result:
    print(i['output']["phonon_bandstructure"]) 
    # get phonon banstructure pymatgen object
```

# Updating failed jobs time limit or execution config
```python
from jobflow_remote.jobs.jobcontroller import JobController

jc = JobController.from_project_name(project_name='YOUR_PROJECT_NAME') # initialize a job controller

job_docs = jc.get_jobs_doc(db_ids='214') # query job docs based on different criteria 
# (Check documentation to see all available options https://github.com/Matgenix/jobflow-remote/blob/967e7c512f230105b1a82c2227fb101d8d4acb3d/src/jobflow_remote/jobs/jobcontroller.py#L467)

# get your existing resources
resources = job_docs[0].resources

# update time limit in the retrieved dict (you can update any other keys like partition/ nodes etc as well)
resources["time"] = '8:00:00'

jc.rerun_job(db_id=job_docs[0].db_id, force=True) # important for jobs that are in failed state to reset them first
jc.set_job_run_properties(db_ids=[job_docs[0].db_id], resources=resources) # this will update the DB entry
```

> IMPORTANT: When you restart VASP calculations, make sure to move the old VASP files somewhere else, 
> because jobflow-remote will restart your calculation in the same directory and that leads to some clash of old and new files.

# Update pre-exsiting job input parameters in the db

```python
# Note that this way is bit involved and you need to find exact structure of your nested db entry based on type of maker used

# Following is an example for failed vasp job where NPAR and ALGO tags in DB entry are updated
from jobflow_remote.jobs.jobcontroller import JobController

jc = JobController.from_project_name(project_name='YOUR_PROJECT_NAME')

job_collection = jc.db.jobs # get jobs collection from mongoDB

for i in job_collection.find({'db_id': '214'}):
    job_dict = i['_id'] # get object id in mongodb (this is used to as filter)
    incar_settings = i['job']['function']['@bound']['input_set_generator']['user_incar_settings'] # get existing user incar settings

incar_settings.update({'NPAR': 2, 'ALGO': 'FAST'}) # now update incar settings here as per requirement
job_collection.update_one({'_id': job_dict}, {'$set': {'job.function.@bound.input_set_generator.user_incar_settings' : incar_settings}})

print(jc.get_jobs_doc(db_ids='214')[0].job.maker.input_set_generator.user_incar_settings) # check if entries are updated
```
> IMPORTANT: When you restart VASP calculations, make sure to move the old VASP files somewhere else, 
> because jobflow-remote will restart your calculation in the same directory and that leads to some clash of old and new files.

# Some useful commands

1. `jf job list` (list jobs in the db)
2. `jf flow list` (list of flows in the db)
3. `jf job info jobid` (provides some info of job like workdir, error info if it failed)
4. `jf flow delete -did db_id` (deletes flow from db)
5. `jf flow -h` or `jf job -h` for checking other options

# Some useful links

1. Check slurm.py for finding different available options you can set for resources dict [here](https://github.com/Matgenix/qtoolkit/tree/develop/src/qtoolkit/io)
2. More details on project config and settings can be found [here](https://matgenix.github.io/jobflow-remote/user/projectconf.html)
3. Details on different setup options [here](https://matgenix.github.io/jobflow-remote/user/install.html)
