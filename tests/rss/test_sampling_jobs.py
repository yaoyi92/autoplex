import os 
os.environ["OMP_NUM_THREADS"] = "1"

from pymatgen.io.ase import AseAtomsAdaptor
from autoplex.data.common.jobs import sampling
from jobflow import run_locally
from ase.io import read
from autoplex.data.common.utils import cur_select, boltzhist_cur, ElementCollection

def test_sampling_cur(test_dir):
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index=':')
    num_of_selection=5
    soap_paras = {'l_max': 3,
                  'n_max': 3,
                  'atom_sigma': 0.5,
                  'cutoff': 3.0,
                  'cutoff_transition_width': 1.0,
                  'zeta': 4.0,
                  'average': True,
                  'species': True,
                 }
    n_species = ElementCollection(atoms).get_number_of_species()
    species_Z = ElementCollection(atoms).get_species_Z()
    descriptor = 'soap l_max=' + str(soap_paras['l_max']) + \
                ' n_max=' + str(soap_paras['n_max']) + \
                ' atom_sigma=' + str(soap_paras['atom_sigma']) + \
                ' cutoff=' + str(soap_paras['cutoff']) + \
                ' n_species=' + str(n_species) + \
                ' species_Z=' + species_Z + \
                ' cutoff_transition_width=' + str(soap_paras['cutoff_transition_width']) + \
                ' average=' + str(soap_paras['average'])

    selected_atoms = cur_select(atoms=atoms, 
                            selected_descriptor=descriptor,
                            kernel_exp=4, 
                            select_nums=num_of_selection, 
                            stochastic=True,
                            random_seed=42)
    
    ref_energies = [-45.44429771,
                    -50.33287125,
                    -29.98566279,
                    -38.71543373,
                    -42.31881099]
    
    energies = [at.info['REF_energy'] for at in selected_atoms]
    
    assert energies == ref_energies


def test_sampling_cur_job(test_dir, memory_jobstore, clean_dir):
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index=':')
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]

    job = sampling(
        selection_method='cur',
        num_of_selection=5,
        bcur_params={'soap_paras': {'l_max': 3,
                                    'n_max': 3,
                                    'atom_sigma': 0.5,
                                    'cutoff': 4.0,
                                    'cutoff_transition_width': 1.0,
                                    'zeta': 4.0,
                                    'average': True,
                                    'species': True,
                                    },
                      },
        structure=structures,
    )
    
    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    selected_atoms = job.output.resolve(memory_jobstore)

    assert len(selected_atoms) == 5

    for atom in selected_atoms:
        assert isinstance(atom, type(structures[0]))



def test_sampling_bcur(test_dir, clean_dir):
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index=':')
    num_of_selection=5
    soap_paras = {'l_max': 3,
                  'n_max': 3,
                  'atom_sigma': 0.5,
                  'cutoff': 3.0,
                  'cutoff_transition_width': 1.0,
                  'zeta': 4.0,
                  'average': True,
                  'species': True,
                 }
    n_species = ElementCollection(atoms).get_number_of_species()
    species_Z = ElementCollection(atoms).get_species_Z()
    descriptor = 'soap l_max=' + str(soap_paras['l_max']) + \
                ' n_max=' + str(soap_paras['n_max']) + \
                ' atom_sigma=' + str(soap_paras['atom_sigma']) + \
                ' cutoff=' + str(soap_paras['cutoff']) + \
                ' n_species=' + str(n_species) + \
                ' species_Z=' + species_Z + \
                ' cutoff_transition_width=' + str(soap_paras['cutoff_transition_width']) + \
                ' average=' + str(soap_paras['average'])

    selected_atoms = boltzhist_cur(atoms=atoms,
                                   isol_es={14: -0.81},
                                   bolt_frac=0.8,
                                   bolt_max_num=3000,
                                   cur_num=num_of_selection,
                                   kernel_exp=4,
                                   kT=0.3,
                                   energy_label='REF_energy',
                                   P=None,
                                   descriptor=descriptor,
                                   random_seed=42,
                                   )
    
    ref_energies = [-78.30403724, 
                    -50.33287125, 
                    -45.44429771, 
                    -105.68052461, 
                    -89.17434151]
    
    energies = [at.info['REF_energy'] for at in selected_atoms]

    assert energies == ref_energies


def test_sampling_bcur_job(test_dir, memory_jobstore, clean_dir):
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index=':')
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    
    job = sampling(selection_method='bcur',
                   num_of_selection=5,
                   bcur_params={'soap_paras': {'l_max': 3,
                                'n_max': 3,
                                'atom_sigma': 0.5,
                                'cutoff': 4.0,
                                'cutoff_transition_width': 1.0,
                                'zeta': 4.0,
                                'average': True,
                                'species': True,
                                },
                                'frac_of_bcur': 0.8,
                                'energy_label': 'REF_energy'
                    },
                   structure=structures,
                   isol_es={14: -0.84696938},
                   random_seed=42)

    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    selected_atoms = job.output.resolve(memory_jobstore)

    assert len(selected_atoms) == 5

    for atom in selected_atoms:
        assert isinstance(atom, type(structures[0]))



def test_sampling_random_job(test_dir, memory_jobstore, clean_dir):
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index=':')
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    
    job = sampling(selection_method='random',
                   num_of_selection=5,
                   structure=structures,
                   random_seed=42)

    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    selected_atoms = job.output.resolve(memory_jobstore)

    assert len(selected_atoms) == 5

    for atom in selected_atoms:
        assert isinstance(atom, type(structures[0]))



def test_sampling_uniform_job(test_dir, memory_jobstore, clean_dir):
    test_files_dir = test_dir / "data/rss.extxyz"
    atoms = read(test_files_dir, index=':')
    structures = [AseAtomsAdaptor.get_structure(atom) for atom in atoms]
    
    job = sampling(selection_method='uniform',
                   num_of_selection=5,
                   structure=structures,
                   random_seed=42)

    response = run_locally(
        job,
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    selected_atoms = job.output.resolve(memory_jobstore)

    assert len(selected_atoms) == 5

    for atom in selected_atoms:
        assert isinstance(atom, type(structures[0]))
