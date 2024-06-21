"""Jobs to create training data for ML potentials."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emmet.core.math import Matrix3D
    from pymatgen.core import Structure

import os
import pickle
from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.constraints import voigt_6_to_full_3x3_stress
from ase.io import read, write
from atomate2.utils.path import strip_hostname
from atomate2.vasp.sets.core import StaticSetGenerator
from atomate2.vasp.jobs.core import StaticMaker
from atomate2.vasp.powerups import update_user_incar_settings
from jobflow.core.job import job
from phonopy.structure.cells import get_supercell
from pymatgen.core.structure import Structure, Lattice
from pymatgen.core import Lattice
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure
from pymatgen.io.vasp.outputs import Vasprun
from jobflow import job, Maker, Response, Flow
from dataclasses import dataclass, field
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import traceback
from typing import Optional, Dict
from custodian.vasp.handlers import (
    FrozenJobErrorHandler,
    IncorrectSmearingHandler,
    LargeSigmaHandler,
    MeshSymmetryErrorHandler,
    NonConvergingErrorHandler,
    PotimErrorHandler,
    StdErrHandler,
    UnconvergedErrorHandler,
    VaspErrorHandler,
)

from autoplex.data.common.utils import (
    mc_rattle,
    random_vary_angle,
    scale_cell,
    std_rattle,
    to_ase_trajectory,
    Species,
    cur_select,
    boltzhist_CUR,
)


@job
def convert_to_extxyz(job_output, pkl_file, config_type, factor):
    """
    Convert data and write extxyt file.

    Parameters
    ----------
    job_output:
        the (static) job output object.
    pkl_file:
        a pickle file.
    config_type: str
            configuration type of the data.
    factor: str
            string of factor to resize cell parameters.

    """
    with open(Path(job_output.dir_name) / Path(pkl_file), "rb") as file:
        traj_obj = pickle.load(file)
    # ForceFieldTaskDocument.from_ase_compatible_result() has no attribute dir_name implemented
    data = to_ase_trajectory(traj_obj=traj_obj)
    data[-1].write("tmp.xyz")
    file = read("tmp.xyz", index=":")
    for i in file:
        virial_list = -voigt_6_to_full_3x3_stress(i.get_stress()) * i.get_volume()
        i.info["REF_virial"] = " ".join(map(str, virial_list.flatten()))
        del i.calc.results["stress"]
        i.arrays["REF_forces"] = i.calc.results["forces"]
        del i.calc.results["forces"]
        i.info["REF_energy"] = i.calc.results["energy"]
        del i.calc.results["energy"]
        i.info["config_type"] = config_type
        i.pbc = True
    write("ref_" + factor + ".extxyz", file, append=True)

    return os.getcwd()


@job
def plot_force_distribution(
    cell_factor: float,
    path,
    x_min: int = 0,
    x_max: int = 5,
    bin_width: float = 0.125,
):
    """
    Plotter for the force distribution.

    Parameters
    ----------
    cell_factor: float
        factor to resize cell parameters.
    x_min: int
        minimum value for the plot x-axis.
    x_max: int
        maximum value for the plot x-axis.
    bin_width: float
        width of the plot bins.

    """
    plt.xlabel("Forces")
    plt.ylabel("Count")
    bins = np.arange(x_min, x_max + bin_width, bin_width)
    plot_total = []

    # TODO split data collection and plotting

    plot_data = []
    with open(path + "/ref_" + str(cell_factor).replace(".", "") + ".extxyz") as file:
        for line in file:
            # Split the line into columns
            columns = line.split()

            # Check if the line has exactly 10 columns
            if len(columns) == 10:
                # Extract the last three columns
                data = columns[-3:]
                norm_data = np.linalg.norm(data, axis=-1)
                plot_data.append(norm_data)

        plt.hist(plot_data, bins=bins, edgecolor="black")
        plt.title(f"Data for factor {cell_factor}")

        plt.savefig("Data_factor_" + str(cell_factor).replace(".", "") + ".png")
        plt.show()

        plot_total += plot_data
    plt.hist(plot_total, bins=bins, edgecolor="black")
    plt.title("Data")

    plt.savefig("Total_data.png")
    plt.show()


@job
def get_supercell_job(structure: Structure, supercell_matrix: Matrix3D):
    """
    Create a job to get the supercell.

    Parameters
    ----------
    structure: Structure
        pymatgen structure object.
    supercell_matrix: Matrix3D
        The matrix to generate the supercell.

    Returns
    -------
    supercell: Structure
        pymatgen structure object.

    """
    supercell = get_supercell(
        unitcell=get_phonopy_structure(structure), supercell_matrix=supercell_matrix
    )
    return get_pmg_structure(supercell)


@job
def generate_randomized_structures(
    structure: Structure,
    supercell_matrix: Matrix3D | None = None,
    distort_type: int = 0,
    n_structures: int = 10,
    volume_scale_factor_range: list[float] | None = None,
    volume_custom_scale_factors: list[float] | None = None,
    min_distance: float = 1.5,
    angle_percentage_scale: float = 10,
    angle_max_attempts: int = 1000,
    rattle_type: int = 0,
    rattle_std: float = 0.01,
    rattle_seed: int = 42,
    rattle_mc_n_iter: int = 10,
    w_angle: list[float] | None = None,
):
    """
    Take in a pymatgen Structure object and generates angle/volume distorted + rattled structures.

    Parameters
    ----------
    structure : Structure.
        Pymatgen structures object.
    supercell_matrix: Matrix3D.
        Matrix for obtaining the supercell.
    distort_type : int.
        0- volume distortion, 1- angle distortion, 2- volume and angle distortion. Default=0.
    n_structures : int.
        Total number of distorted structures to be generated.
        Must be provided if distorting volume without specifying a range, or if distorting angles.
        Default=10.
    volume_scale_factor_range : list[float]
        [min, max] of volume scale factors.
        e.g. [0.90, 1.10] will distort volume +-10%.
    volume_custom_scale_factors : list[float]
        Specify explicit scale factors (if range is not specified).
        If None, will default to [0.90, 0.95, 0.98, 0.99, 1.01, 1.02, 1.05, 1.10].
    min_distance: float
        Minimum separation allowed between any two atoms.
        Default= 1.5A.
    angle_percentage_scale: float
        Angle scaling factor.
        Default= 10 will randomly distort angles by +-10% of original value.
    angle_max_attempts: int.
        Maximum number of attempts to distort structure before aborting.
        Default=1000.
    w_angle: list[float]
        List of angle indices to be changed i.e. 0=alpha, 1=beta, 2=gamma.
        Default= [0, 1, 2].
    rattle_type: int.
        0- standard rattling, 1- Monte-Carlo rattling. Default=0.
    rattle_std: float.
        Rattle amplitude (standard deviation in normal distribution).
        Default=0.01.
        Note that for MC rattling, displacements generated will roughly be
        rattle_mc_n_iter**0.5 * rattle_std for small values of n_iter.
    rattle_seed: int.
        Seed for setting up NumPy random state from which random numbers are generated.
        Default=42.
    rattle_mc_n_iter: int.
        Number of Monte Carlo iterations.
        Larger number of iterations will generate larger displacements.
        Default=10.

    Returns
    -------
    Response.output.
        Volume or angle-distorted structures with rattled atoms.
    """
    if supercell_matrix is None:
        supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    supercell = get_supercell(
        unitcell=get_phonopy_structure(structure),
        supercell_matrix=supercell_matrix,
    )
    structure = get_pmg_structure(supercell)
    # distort cells by volume or angle
    if distort_type == 0:
        distorted_cells = scale_cell(
            structure=structure,
            volume_scale_factor_range=volume_scale_factor_range,
            n_structures=n_structures,
            volume_custom_scale_factors=volume_custom_scale_factors,
        )
    elif distort_type == 1:
        distorted_cells = random_vary_angle(
            structure=structure,
            min_distance=min_distance,
            angle_percentage_scale=angle_percentage_scale,
            w_angle=w_angle,
            n_structures=n_structures,
            angle_max_attempts=angle_max_attempts,
        )
    elif distort_type == 2:
        initial_distorted_cells = scale_cell(
            structure=structure,
            volume_scale_factor_range=volume_scale_factor_range,
            n_structures=n_structures,
            volume_custom_scale_factors=volume_custom_scale_factors,
        )
        distorted_cells = []
        for cell in initial_distorted_cells:
            distorted_cell = random_vary_angle(
                structure=cell,
                min_distance=min_distance,
                angle_percentage_scale=angle_percentage_scale,
                w_angle=w_angle,
                n_structures=1,
                angle_max_attempts=angle_max_attempts,
            )
            distorted_cells.append(distorted_cell)
        distorted_cells = list(chain.from_iterable(distorted_cells))
    else:
        raise TypeError("distort_type is not recognised")

    # distorted_cells=list(chain.from_iterable(distorted_cells))

    # rattle cells by standard or mc
    rattled_cells = (
        [
            std_rattle(
                structure=cell,
                n_structures=1,
                rattle_std=rattle_std,
                rattle_seed=rattle_seed,
            )
            for cell in distorted_cells
        ]
        if rattle_type == 0
        else [
            mc_rattle(
                structure=cell,
                n_structures=1,
                rattle_std=rattle_std,
                min_distance=min_distance,
                rattle_seed=rattle_seed,
                rattle_mc_n_iter=rattle_mc_n_iter,
            )
            for cell in distorted_cells
        ]
        if rattle_type == 1
        else None
    )

    if rattled_cells is None:
        raise TypeError("rattle_type is not recognized")

    return list(chain.from_iterable(rattled_cells))


@job
def Sampling(selection_method: str = None,
             num_of_selection : int = 5,
             bcur_params: Optional[Dict] = None,
             dir: str = None, 
             structure: list[Structure] = None, 
             traj_info: list = None, 
             isol_es: Optional[Dict] = None):
    """
    Job to sample training configurations from trajs of MD/RSS.
    
    Parameters
    ----------
    selection_method : str, optional
        Method for selecting samples. Options include:
        - 'cur': Pure CUR selection.
        - 'boltzhist_CUR': Boltzmann flat histogram in enthalpy, then CUR.
        - 'random': Random selection.
        - 'uniform': Uniform selection. Default is None. If None, then default to random.
    
    num_of_selection : int, optional
        Number of selections to be made. Default is 5.
    
    bcur_params : dict, optional
        Parameters for Boltzmann CUR selection. The default dictionary includes:
        - 'soap_paras': SOAP descriptor parameters:
            - 'l_max': int, Maximum degree of spherical harmonics (default 8).
            - 'n_max': int, Maximum number of radial basis functions (default 8).
            - 'atom_sigma': float, Width of Gaussian smearing (default 0.75).
            - 'cutoff': float, Radial cutoff distance (default 5.5).
            - 'cutoff_transition_width': float, Width of the transition region (default 1.0).
            - 'zeta': float, Exponent for dot-product SOAP kernel (default 4.0).
            - 'average': bool, Whether to average the SOAP vectors (default True).
            - 'species': bool, Whether to consider species information (default True).
        - 'kT': float, Temperature in eV for Boltzmann weighting (default 0.3).
        - 'frac_of_bcur': float, Fraction of Boltzmann CUR selections (default 0.1).
        - 'bolt_max_num': int, Maximum number of Boltzmann selections (default 3000).
        - 'kernel_exp': float, Exponent for the kernel (default 4.0).
        - 'energy_label': str, Label for the energy data (default 'energy').
    
    dir : str, optional
        Directory containing trajectory files for MD/RSS simulations. Default is None.
    
    structure : list[Structure], optional
        List of structures for sampling. Default is None.
    
    traj_info : list, optional
        List of dictionaries containing trajectory information. Each dictionary should 
        have keys 'traj_path' and 'pressure'. Default is None.
    
    isol_es : dict, optional
        Dictionary of isolated energy values for species. Required for 'boltzhist_CUR' 
        selection method. Default is None.

    Returns
    -------
    list of ase.Atoms
        The selected atoms. These are copies of the atoms in the input list.
    """

    default_bcur_params = {
        'soap_paras': {
            'l_max': 8,
            'n_max': 8,
            'atom_sigma': 0.75,
            'cutoff': 5.5,
            'cutoff_transition_width': 1.0,
            'zeta': 4.0,
            'average': True,
            'species': True,
        },
        'kT': 0.3,
        'frac_of_bcur': 0.1,
        'bolt_max_num': 3000,
        'kernel_exp': 4.0,
        'energy_label': 'energy'
    }

    if bcur_params is not None:
        default_bcur_params.update(bcur_params)
    
    bcur_params = default_bcur_params
        
    if dir is not None:
        atoms = read(dir, index=':')

    elif structure is not None:
        atoms = [AseAtomsAdaptor().get_atoms(at) for at in structure]
        
    else:  
        atoms = []
        pressures = []
        for traj in traj_info:
            if traj is not None:
                print('traj:', traj)
                at = read(traj['traj_path'],index=':')
                atoms.extend(at)
                pressure = [traj['pressure']] * len(at)
                pressures.extend(pressure)

    if selection_method == 'cur' or selection_method == 'boltzhist_CUR':

        n_species = Species(atoms).get_number_of_species()
        species_Z = Species(atoms).get_species_Z()

        soap_paras = bcur_params['soap_paras']
        descriptor = 'soap l_max=' + str(soap_paras['l_max']) + \
                    ' n_max=' + str(soap_paras['n_max']) + \
                    ' atom_sigma=' + str(soap_paras['atom_sigma']) + \
                    ' cutoff=' + str(soap_paras['cutoff']) + \
                    ' n_species=' + str(n_species) + \
                    ' species_Z=' + species_Z + \
                    ' cutoff_transition_width=' + str(soap_paras['cutoff_transition_width']) + \
                    ' average =' + str(soap_paras['average'])

        if selection_method == 'cur':
    
            selected_atoms = cur_select(atoms=atoms, 
                                        selected_descriptor=descriptor,
                                        kernel_exp=bcur_params['kernel_exp'], 
                                        select_nums=num_of_selection, 
                                        stochastic=True)


        elif selection_method == 'boltzhist_CUR':

            isol_es = {int(k): v for k, v in isol_es.items()}

            selected_atoms = boltzhist_CUR(atoms=atoms,
                                        isol_es=isol_es,
                                        bolt_frac=bcur_params['frac_of_bcur'], 
                                        bolt_max_num=bcur_params['bolt_max_num'],
                                        cur_num=num_of_selection, 
                                        kernel_exp=bcur_params['kernel_exp'], 
                                        kT=bcur_params['kT'], 
                                        energy_label=bcur_params['energy_label'],
                                        P=pressures,
                                        descriptor=descriptor
                                        )

        selected_atoms = [AseAtomsAdaptor().get_structure(at) for at in selected_atoms]

        return selected_atoms
    
    elif selection_method == None or selection_method == 'random':
        
        structure = [AseAtomsAdaptor().get_structure(at) for at in atoms]

        try: 
            selection = np.random.choice(0, len(structure), num_of_selection)
            selected_atoms = [at for i, at in enumerate(structure) if i in selection]

        except:
            print('[log] The number of selected structures must be less than the total!')
            traceback.print_exc()

        return selected_atoms
    
    elif selection_method == 'uniform':

        try: 
            indices = np.linspace(0, len(atoms) - 1, num_of_selection, dtype=int)
            structure = [AseAtomsAdaptor().get_structure(at) for at in atoms]
            selected_atoms = [structure[idx] for idx in indices]

        except:
            print('[log] The number of selected structures must be less than the total!')
            traceback.print_exc()

        return selected_atoms
        

@job
def VASP_static(structures: list[Structure] | None = None, 
                config_types: list[str] | None = None, 
                isolated_atom: bool = False,
                isolated_species: list[str] | None = None,
                e0_spin: bool = False,
                dimer: bool = False,
                dimer_species: list[str] | None = None,
                dimer_range: list[float] = [1.0, 5.0],
                dimer_num: int = 21,
                custom_set: dict | None = None
                ):
    """
    Jobflow to set up and run VASP static calculations for input structures, 
    including bulk, isolated atoms, and dimers. It supports custom VASP input 
    parameters and error handlers.

    Parameters
    ----------
    structures : list[Structure], optional
        List of structures for which to run the VASP static calculations. If None, 
        no bulk calculations will be performed. Default is None.
    
    config_types : list[str], optional
        List of configuration types corresponding to the structures. If provided, 
        should have the same length as the 'structures' list. If None, defaults 
        to 'bulk'. Default is None.
    
    isolated_atom : bool, optional
        Whether to perform calculations for isolated atoms. Default is False.
    
    isolated_species : list[str], optional
        List of species for which to perform isolated atom calculations. If None, 
        species will be automatically derived from the 'structures' list. Default is None.
    
    e0_spin : bool, optional
        Whether to include spin polarization in isolated atom and dimer calculations. 
        Default is False.
    
    dimer : bool, optional
        Whether to perform calculations for dimers. Default is False.
    
    dimer_species : list[str], optional
        List of species for which to perform dimer calculations. If None, species 
        will be derived from the 'structures' list. Default is None.
    
    dimer_range : list[float], optional
        Range of distances for dimer calculations. Default is [0.8, 4.8].
    
    dimer_num : int, optional
        Number of different distances to consider for dimer calculations. Default is 22.
    
    custom_set : dict, optional
        Dictionary of custom VASP input parameters. If provided, will update the 
        default parameters. Default is None.

    Returns
    -------
    Response
        A Response object containing the VASP jobs and the directories where 
        the calculations were set up.
    """
    
    job_list = []

    dirs = {'dirs_of_vasp': [], 'config_type': []}

    default_custom_set = {
        "ADDGRID": "True", 
        "ENCUT": 520,
        "EDIFF": 1E-06,
        "ISMEAR": 0,
        "SIGMA": 0.01,
        "PREC": "Accurate",
        "ISYM": None,
        "KSPACING": 0.2,
        "NPAR": 8,
        "LWAVE": "False",
        "LCHARG": "False",
        "ENAUG": None,
        "GGA": None,
        "ISPIN": None,
        "LAECHG": None,
        "LELF": None,
        "LORBIT": None,
        "LVTOT": None,
        "NSW": None,
        "SYMPREC": None,
        "NELM": 100,
        "LMAXMIX": None,
        "LASPH": None,
        "AMIN": None,
    }

    if custom_set is not None:
        default_custom_set.update(custom_set)
    
    custom_set = default_custom_set

    custom_handlers = (VaspErrorHandler(),
                       MeshSymmetryErrorHandler(),
                       UnconvergedErrorHandler(),
                       NonConvergingErrorHandler(),
                       PotimErrorHandler(),
                       FrozenJobErrorHandler(),
                       StdErrHandler(),
                       LargeSigmaHandler(),
                       IncorrectSmearingHandler(),
                       )

    st_m = StaticMaker(
        input_set_generator = StaticSetGenerator(
        user_incar_settings = custom_set),
        run_vasp_kwargs = {"handlers": custom_handlers},
    )

    if structures:
        for idx, struct in enumerate(structures):
            static_job = st_m.make(structure = struct)
            dirs['dirs_of_vasp'].append(static_job.output.dir_name)
            if config_types:
                dirs['config_type'].append(config_types[idx])
            else:
                dirs['config_type'].append('bulk')
            job_list.append(static_job)


    if isolated_atom:
        try:
            if isolated_species is not None:
                
                syms = isolated_species

            elif (isolated_species is None) and (structures is not None):
                
                # Get the species from the database        
                atoms = [AseAtomsAdaptor().get_atoms(at) for at in structures]
                syms = Species(atoms).get_species()
        
            for sym in syms:
                lattice = Lattice.orthorhombic(20.0, 20.5, 21.0)
                isolated_atom_struct = Structure(lattice,[sym], [[0.0, 0.0, 0.0]])
                static_job = st_m.make(structure = isolated_atom_struct)
                static_job = update_user_incar_settings(static_job, {"KSPACING": 2.0})

                if e0_spin:
                    static_job = update_user_incar_settings(static_job, {"ISPIN": 2})
                
                dirs['dirs_of_vasp'].append(static_job.output.dir_name)
                dirs['config_type'].append('isolated_atom')
                job_list.append(static_job)
 
        except: 
            raise ValueError('[log] Unknown species of isolated atoms!') 
         
    if dimer:
        try:
            if dimer_species is not None:        
                dimer_syms = dimer_species
            elif (dimer_species is None) and (structures is not None):
                # Get the species from the database        
                atoms = [AseAtomsAdaptor().get_atoms(at) for at in structures]
                dimer_syms = Species(atoms).get_species()
            pairs_list = Species(atoms).find_element_pairs(dimer_syms)
            for pair in pairs_list:
                for dimer_i in range(dimer_num):
                    dimer_distance = dimer_range[0] + (dimer_range[1] - dimer_range[0]) * \
                                     float(dimer_i) / float(dimer_num - 1 + 0.000000000001)
                    
                    lattice = Lattice.orthorhombic(15.0, 15.5, 16.0)
                    dimer_struct = Structure(lattice,
                                            [pair[0], pair[1]], 
                                            [[0.0, 0.0, 0.0], 
                                             [dimer_distance, 0.0, 0.0]],
                                            coords_are_cartesian=True)
            
                    static_job = st_m.make(structure = dimer_struct)
                    static_job = update_user_incar_settings(static_job, {"KSPACING": 2.0})

                    if e0_spin:
                        static_job = update_user_incar_settings(static_job, {"ISPIN": 2})

                    dirs['dirs_of_vasp'].append(static_job.output.dir_name)
                    dirs['config_type'].append('dimer')
                    job_list.append(static_job)
                    
        except:
            raise ValueError('[log] Unknown atom types in dimers!') 
        
    return Response(replace=Flow(job_list), output=dirs)
    

@job
def VASP_collect_data(vasp_ref_file: str = 'vasp_ref.extxyz',
                      gap_rss_group: str = 'RSS',
                      vasp_dirs: Optional[Dict[str, list]] = None):
    
    """
    Collects VASP data from specified directories.

    Parameters
    ----------
    vasp_ref_file : str, optional
        Reference file for VASP data. Default is 'vasp_ref.extxyz'.
    
    gap_rss_group : str, optional
        Group name for GAP RSS. Default is 'RSS'.
    
    vasp_dirs : dict, mandatory
        Dictionary containing VASP directories and configuration types. Should have keys:
        - 'dirs_of_vasp': List of directories containing VASP data.
        - 'config_type': List of configuration types corresponding to each directory.
    
    Returns
    -------
    dict
        A dictionary containing:
        - 'vasp_ref_dir': Directory of the VASP reference file.
        - 'isol_es': Isolated energy values.
    """
        
    if vasp_dirs is None:
        raise ValueError("vasp_dirs must be provided and should contain 'dirs_of_vasp' and 'config_type' keys.")
    
    if 'dirs_of_vasp' not in vasp_dirs or 'config_type' not in vasp_dirs:
        raise ValueError("vasp_dirs must contain 'dirs_of_vasp' and 'config_type' keys.")

    dirs = [safe_strip_hostname(value) for value in vasp_dirs['dirs_of_vasp']]
    config_types = vasp_dirs['config_type']

    print('[log] Attempting collecting VASP...', flush=True)

    if dirs == None:
        raise ValueError('[log] dft_dir must be specified if collect_vasp is True')
    
    failed_count = 0
    atoms = []
    isol_es = {}

    for i, val in enumerate(dirs):

        if os.path.exists(os.path.join(val, 'vasprun.xml.gz')): 
            
            try:
                converged = check_convergence_vasp(os.path.join(val, 'vasprun.xml.gz'))

                if converged:
                    at = read(os.path.join(val, 'vasprun.xml.gz'), index=':')
                    for at_i in at:
                        virial_list = -voigt_6_to_full_3x3_stress(at_i.get_stress()) * at_i.get_volume()
                        at_i.info['REF_virial'] = ' '.join(map(str, virial_list.flatten()))
                        del at_i.calc.results['stress']
                        at_i.arrays['REF_forces'] = at_i.calc.results['forces']
                        del at_i.calc.results['forces']
                        at_i.info['REF_energy'] = at_i.calc.results['free_energy']
                        del at_i.calc.results['energy']
                        del at_i.calc.results['free_energy']
                        atoms.append(at_i)
                        at_i.info['config_type'] = config_types[i]
                        if at_i.info['config_type'] != 'dimer' and at_i.info['config_type'] != 'isolated_atom':
                            at_i.pbc=True
                            at_i.info['gap_rss_group']= gap_rss_group
                        else:
                            at_i.info['gap_rss_nonperiodic'] = 'T'

                        if at_i.info['config_type'] == 'isolated_atom':
                            at_ids = at_i.get_atomic_numbers()
                            # array_key = at_ids.tostring()
                            isol_es[int(at_ids[0])] = at_i.info['REF_energy']
            
            except:
                print('[log] Failed to collect number', i)
                failed_count += 1
                traceback.print_exc()
    
    print('[log] Total %d structures from VASP are exactly collected.' % len(atoms))
    
    write(vasp_ref_file, 
          atoms, 
          format='extxyz',
          parallel=False)

    dir_path = Path.cwd()

    vasp_ref_dir = os.path.join(dir_path, vasp_ref_file)

    return {'vasp_ref_dir':vasp_ref_dir, 'isol_es':isol_es}


def check_convergence_vasp(file):

    """
    Check if VASP calculation has converged.
    True if a run is converged both ionically and electronically.

    """

    vasprun = Vasprun(file)
    converged_e = vasprun.converged_electronic
    converged_i = vasprun.converged_ionic

    if converged_e and converged_i: 
        return True
    
    else: 
        return False
    

def safe_strip_hostname(value):
    """
    Strips the hostname from a given path or URL.

    Parameters
    ----------
    value : str
        The path or URL from which to strip the hostname.

    Returns
    -------
    Optional[str]
        The path or URL without the hostname if the operation is successful, 
        otherwise None.
    """
    
    try:
        return strip_hostname(value)
    except Exception as e:
        print(f"Error processing '{value}': {e}")
        return None