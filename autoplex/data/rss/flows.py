from __future__ import annotations

from jobflow import run_locally, Flow
from jobflow import Response, job
from autoplex.data.rss.jobs import RandomizedStructure, do_rss
from autoplex.data.common.jobs import Sampling, VASP_static, VASP_collect_data
from autoplex.fitting.common.jobs import data_preprocessing, machine_learning_fit
from typing import List, Optional, Dict, Any


@job
def initial_RSS(struct_number: int = 10000,
                tag: str = 'GeSb2Te4',
                selection_method: str = 'cur',
                num_of_selection: int = 100,
                vasp_ref_file: str = 'vasp_ref.extxyz',
                gap_rss_group: str = 'initial',
                split_ratio: float = 0.1,
                regularization: bool = True,
                distillation: bool = True,
                f_max: float = 200,
                pre_database_dir: Optional[str] = None,
                mlip_type: str = 'GAP',
                e0_spin: bool = False,
                isolated_atom: bool = True,
                dimer: bool = True,
                kwargs: Optional[Dict[str, Any]] = None):
    """
    Initial Random Structure Searching (RSS) workflow.
    
    The workflow consists of the following jobs:
    
    1. **Generates randomized structures**:
       - Uses the `RandomizedStructure` class to generate structures.
    
    2. **Samples a subset of the generated structures using CUR selection**:
       - Uses the `Sampling` function to select structures from the generated ones.
    
    3. **Runs static VASP calculations on the sampled structures**:
       - Uses the `VASP_static` function to perform static calculations, including calculations for isolated atoms and dimers.
    
    4. **Collects VASP calculation data**:
       - Uses the `VASP_collect_data` function to collect the data from the VASP calculations.
    
    5. **Preprocesses the data for machine learning**:
       - Uses the `data_preprocessing` function to preprocess the data, including splitting the data, applying regularization, and distillation.
    
    6. **Fits a machine learning interatomic potential (MLIP)**:
       - Uses the `machine_learning_fit` function to fit a MLIP using the preprocessed data.

    Parameters
    ----------
    struct_number : int, optional
        Number of structures to generate. Default is 10000.
    
    tag : str, optional
        Tag for the generated structures. Default is 'GeSb2Te4'.
    
    selection_method : str, optional
        Method for selecting structures. Default is 'cur'.
    
    num_of_selection : int, optional
        Number of structures to select. Default is 100.
    
    vasp_ref_file : str, optional
        Reference file for VASP data. Default is 'vasp_ref.extxyz'.
    
    gap_rss_group : str, optional
        Group name for GAP RSS. Default is 'initial'.
    
    split_ratio : float, optional
        Ratio for data splitting. Default is 0.1.
    
    regularization : bool, optional
        Whether to apply regularization. Default is True.
    
    distillation : bool, optional
        Whether to apply distillation. Default is True.
    
    f_max : float, optional
        Maximum force value. Default is 200.
    
    pre_database_dir : str, optional
        Directory for the preprocessed database. Default is None.
    
    mlip_type : str, optional
        Type of MLIP to fit. Default is 'GAP'.
    
    e0_spin : bool, optional
        Whether to include spin polarization in calculations. Default is False.
    
    isolated_atom : bool, optional
        Whether to include isolated atom calculations. Default is True.
    
    dimer : bool, optional
        Whether to include dimer calculations. Default is True.
    
    kwargs : dict, optional
        Additional arguments for the machine learning fit. Default is None.

    Returns
    -------
    Response
        A jobflow Response object containing a Flow of jobs and the output dictionary.
    
    Output
    ------
    - test_error: float
        The test error of the fitted MLIP.
    - pre_database_dir: str
        The directory of the preprocessed database.
    - mlip_path: str
        The path to the fitted MLIP.
    - isol_es: dict
        The isolated energy values.
    - current_iter: int
        The current iteration index, set to 0.
    - kt: float
        The value of kT, set to 0.6.
    """
    
    job1 = RandomizedStructure(struct_number=struct_number, tag=tag).make()
    job2 = Sampling(selection_method=selection_method, num_of_selection=num_of_selection, dir=job1.output)
    job3 = VASP_static(structures=job2.output, e0_spin=e0_spin, isolated_atom=isolated_atom, dimer=dimer)
    job4 = VASP_collect_data(vasp_ref_file=vasp_ref_file, gap_rss_group=gap_rss_group, vasp_dirs=job3.output)
    job5 = data_preprocessing(split_ratio=split_ratio, regularization=regularization, distillation=distillation, f_max=f_max, vasp_ref_dir=job4.output['vasp_ref_dir'], pre_database_dir=pre_database_dir)
    job6 = machine_learning_fit(database_dir=job5.output, isol_es=job4.output['isol_es'], mlip_type=mlip_type, kwargs=kwargs)
    job_list = [job1, job2, job3, job4, job5, job6]

    return Response(
        replace=Flow(job_list),
        output={
            'test_error': job6.output['test_error'],
            'pre_database_dir': job5.output,
            'mlip_path': job6.output['mlip_path'],
            'isol_es': job4.output['isol_es'],
            'current_iter': 0,
            'kt': 0.6
        },
    )

                            
@job
def do_RSS_iterations(input: Dict[str, Optional[Any]] = {'test_error': None,
                                                         'pre_database_dir': None,
                                                         'mlip_path': None,
                                                         'isol_es': None,
                                                         'current_iter': None,
                                                         'kt': 0.6},
                      struct_number: int = 10000,
                      tag: str = 'GeSb2Te4',
                      selection_method: str = 'cur',
                      num_of_selection: int = 1000,
                      mlip_type: str = 'GAP',
                      vasp_ref_file: str = 'vasp_ref.extxyz',
                      gap_rss_group: str = 'rss',
                      split_ratio: float = 0.1,
                      regularization: bool = True,
                      distillation: bool = True,
                      f_max: float = 80,
                      stop_criterion: float = 0.01,
                      max_iteration_number: int = 9,
                      kwargs: Optional[Dict[str, Any]] = None):
    """

    This workflow performs iterative RSS to improve the accuracy of a MLIP until a stopping criterion is met or a maximum number of iterations is reached. Each iteration involves generating new structures, sampling, running VASP calculations, collecting data, preprocessing data, and fitting a new MLIP.

    Parameters
    ----------
    input : dict, optional
        Dictionary containing the input parameters for the iteration. Default is:
        {
            'test_error': None,
            'pre_database_dir': None,
            'mlip_path': None,
            'isol_es': None,
            'current_iter': None,
            'kt': 0.6
        }.
    
    struct_number : int, optional
        Number of structures to generate. Default is 10000.
    
    tag : str, optional
        Tag for the generated structures. Default is 'GeSb2Te4'.
    
    selection_method : str, optional
        Method for selecting structures. Default is 'cur'.
    
    num_of_selection : int, optional
        Number of structures to select. Default is 1000.
    
    mlip_type : str, optional
        Type of MLIP to fit. Default is 'GAP'.
    
    vasp_ref_file : str, optional
        Reference file for VASP data. Default is 'vasp_ref.extxyz'.
    
    gap_rss_group : str, optional
        Group name for GAP RSS. Default is 'rss'.
    
    split_ratio : float, optional
        Ratio for data splitting. Default is 0.1.
    
    regularization : bool, optional
        Whether to apply regularization. Default is True.
    
    distillation : bool, optional
        Whether to apply distillation. Default is True.
    
    f_max : float, optional
        Maximum force value. Default is 80.
    
    stop_criterion : float, optional
        The stopping criterion for the test error. Default is 0.01.
    
    max_iteration_number : int, optional
        The maximum number of iterations to perform. Default is 9.
    
    kwargs : dict, optional
        Additional arguments for the machine learning fit. Default is None.

    Returns
    -------
    Response
        A jobflow Response object containing the detour jobs and the output dictionary.
    
    Output
    ------
    The output of the final iteration or the input if the stopping criterion is met or the maximum number of iterations is reached.

    """

    if input['test_error'] is not None and input['test_error'] > stop_criterion and input['current_iter'] < max_iteration_number:
        if input['kt'] > 0.15:
            kt = input['kt'] - 0.1
        else:
            kt = 0.1
        print('kt:', kt)
        current_iter = input['current_iter'] + 1
        print('Current iter index:', current_iter)
        print(f'The error of {current_iter}th iteration:', input['test_error'])

        flag1 = RandomizedStructure(struct_number=struct_number, tag=tag).make()
        flag2 = Sampling(selection_method=selection_method, num_of_selection=num_of_selection, dir=flag1.output)
        flag3 = do_rss(mlip_type=mlip_type, iteration_index=f'{current_iter}th', mlip_path=input['mlip_path'], structure=flag2.output)
        flag4 = Sampling(selection_method='boltzhist_CUR', num_of_selection=100, dir=flag3.output, isol_es=input["isol_es"])
        flag5 = VASP_static(structures=flag4.output, isolated_atom=False, dimer=False)
        flag6 = VASP_collect_data(vasp_ref_file=vasp_ref_file, gap_rss_group=gap_rss_group, vasp_dirs=flag5.output)
        flag7 = data_preprocessing(split_ratio=split_ratio, regularization=regularization, distillation=distillation, f_max=f_max, vasp_ref_dir=flag6.output['vasp_ref_dir'], pre_database_dir=input['pre_database_dir'])
        flag8 = machine_learning_fit(database_dir=flag7.output, isol_es=input["isol_es"], mlip_type=mlip_type, kwargs=kwargs)

        flag9 = do_RSS_iterations(input={'test_error': flag8.output['test_error'],
                                         'pre_database_dir': flag7.output,
                                         'mlip_path': flag8.output['mlip_path'],
                                         'isol_es': input["isol_es"],
                                         'current_iter': current_iter,
                                         'kt': kt},
                                 )
        
        job_list = [flag1, flag2, flag3, flag4, flag5, flag6, flag7, flag8, flag9]

        return Response(detour=job_list, output=flag9.output)
    
    else:

        return Response(output=input)
    
    