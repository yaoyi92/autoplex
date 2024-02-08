import ase.io
import os
import numpy as np
from ase.neighborlist import NeighborList, natural_cutoffs
from autoplex.fitting.utilities import energy_remain
from ase.data import chemical_symbols
import pandas as pd
import yaml


########## fitting using GAP ##########

def calculate_delta(atoms_db, e_name):
    at_ids = [atom.get_atomic_numbers() for atom in atoms_db]
    isol_es = {atom.get_atomic_numbers()[0]: atom.info[e_name] for atom in atoms_db if 'config_type' in atom.info and 'isol' in atom.info['config_type']}
    es_visol = np.array([(atom.info[e_name] - sum([isol_es[j] for j in at_ids[ct]])) / len(atom) for ct, atom in enumerate(atoms_db)])
    es_var = np.var(es_visol)
    avg_neigh = np.mean([compute_average_coordination(atom) for atom in atoms_db])
    return es_var / avg_neigh


def compute_average_coordination(atoms):
    cutoffs = natural_cutoffs(atoms)
    neighbor_list = NeighborList(cutoffs, self_interaction=False, bothways=True)
    neighbor_list.update(atoms)
    total_coordination = sum(len(neighbor_list.get_neighbors(index)[0]) for index in range(len(atoms)))
    return total_coordination / len(atoms)


def run_command(command):
    os.system(command)


def gap_fitting(dir, two_body=True, three_body=False, soap=True):
    db_atoms = ase.io.read(os.path.join(dir, 'train.extxyz'), index=':')
    train_data_path = os.path.join(dir, 'train_with_sigma.extxyz')
    test_data_path = os.path.join(dir, 'test.extxyz')

    parameters = []

    if two_body:
        delta_2b = calculate_delta(db_atoms, 'REF_energy')
        parameters.append(f'distance_Nb order=2 compact_clusters=T cutoff=5.0 add_species=T covariance_type=ARD_SE theta_uniform=0.5 sparse_method=uniform n_sparse=15 delta={delta_2b} f0=0.0')

        gap_command = 'export OMP_NUM_THREADS=32 && gap_fit energy_parameter_name=REF_energy force_parameter_name=REF_forces virial_parameter_name=REF_virial do_copy_at_file=F at_file={} gap={{ {} }} default_sigma={{0.0001 0.05 0.05 0}} sparse_jitter=1.0e-10 gp_file=gap_file.xml'.format(train_data_path, ' : '.join(parameters))
        run_command(gap_command)

        quip_command = "export OMP_NUM_THREADS=32 && quip E=T F=T atoms_filename={} param_filename=gap_file.xml | grep AT | sed 's/AT//' > quip_train.extxyz".format(train_data_path)
        run_command(quip_command)

    if three_body:
        delta_3b = energy_remain('quip_train.extxyz')
        parameters.append(f'distance_Nb order=3 compact_clusters=T cutoff=3.25 covariance_type=ard_se theta_uniform=1.0 sparse_method=uniform n_sparse=100 f0=0.0 add_species=T delta={delta_3b}')

        gap_command = 'export OMP_NUM_THREADS=32 && gap_fit energy_parameter_name=REF_energy force_parameter_name=REF_forces virial_parameter_name=REF_virial do_copy_at_file=F at_file={} gap={{ {} }} default_sigma={{0.0001 0.05 0.05 0}} sparse_jitter=1.0e-10 gp_file=gap_file.xml'.format(train_data_path, ' : '.join(parameters))
        run_command(gap_command)

        quip_command = "export OMP_NUM_THREADS=32 && quip E=T F=T atoms_filename={} param_filename=gap_file.xml | grep AT | sed 's/AT//' > quip_train.extxyz".format(train_data_path)
        run_command(quip_command)

    if soap:
        delta_soap = energy_remain('quip_train.extxyz') if two_body or three_body else 1
        parameters.append(f'soap l_max=6 n_max=10 atom_sigma=0.5 zeta=4 cutoff=5.0 cutoff_transition_width=0.5 central_weight=1.0 n_sparse=1000 f0=0.0 R_mix=T Z_mix=T K=5 covariance_type=dot_product sparse_method=cur_points add_species=T delta={delta_soap}')

        gap_command = 'export OMP_NUM_THREADS=32 && gap_fit energy_parameter_name=REF_energy force_parameter_name=REF_forces virial_parameter_name=REF_virial do_copy_at_file=F at_file={} gap={{ {} }} default_sigma={{0.0001 0.05 0.05 0}} sparse_jitter=1.0e-10 gp_file=gap_file.xml'.format(train_data_path, ' : '.join(parameters))
        run_command(gap_command)

        quip_command = "export OMP_NUM_THREADS=32 && quip E=T F=T atoms_filename={} param_filename=gap_file.xml | grep AT | sed 's/AT//' > quip_train.extxyz".format(train_data_path)
        run_command(quip_command)

    # Calculate training error
    train_error = energy_remain('quip_train.extxyz')
    print('Training error of MLIP (eV/at.):', train_error)

    # Calculate testing error
    quip_command = "export OMP_NUM_THREADS=32 && quip E=T F=T atoms_filename={} param_filename=gap_file.xml | grep AT | sed 's/AT//' > quip_test.extxyz".format(test_data_path)
    run_command(quip_command)
    test_error = energy_remain('quip_test.extxyz')
    print('Testing error of MLIP (eV/at.):', test_error)

    return train_error, test_error


########## fitting using Julia-ACE ##########

def ace_fitting(dir=None, 
                energy_name="REF_energy", 
                force_name="REF_forces", 
                virial_name="REF_virial",
                order=4,
                totaldegree=16,
                cutoff=5.0,
                solver='BLR',
                isol_es=None,
                num_of_threads=128):

    train_atoms = ase.io.read('train.extxyz', index=':')

    isol_es_update = {}

    if isol_es:
        for e_num, e_energy in isol_es.items():
            isol_es_update[chemical_symbols[int(e_num)]] = e_energy
    else:
        raise ValueError("isol_es is empty or not defined!")

    formatted_isol_es = '[' + ', '.join([f":{key} => {value}" for key, value in isol_es_update.items()]) + ']'
    formatted_species = '[' + ', '.join([f":{key}" for key, value in isol_es_update.items()]) + ']'

    train_ace = [at for at in train_atoms if 'isolated_atom' not in at.info['config_type']]
    ase.io.write('train_ace.extxyz', train_ace, format='extxyz')

    ace_text = f'''using ACEpotentials
using LinearAlgebra: norm, Diagonal
using CSV, DataFrames
using Distributed
addprocs({num_of_threads-1}, exeflags="--project=$(Base.active_project())")
@everywhere using ACEpotentials

data_file = "train_ace.extxyz"
data = read_extxyz(data_file)
test_data_file = "test.extxyz"
test_data = read_extxyz(test_data_file)
data_keys = (energy_key = "{energy_name}", force_key = "{force_name}", virial_key = "{virial_name}")

model = acemodel(elements={formatted_species}, 
                order={order}, 
                totaldegree={totaldegree}, 
                rcut={cutoff},
                Eref={formatted_isol_es})

weights = Dict(
            "bulk" => Dict("E" => 10.0, "F" => 1.0 , "V" => 1.0 ),
            "initial" => Dict("E" => 5.0, "F" => 0.5 , "V" => 0.5 ),
            "dimer" => Dict("E" => 5.0, "F" => 0.5 , "V" => 0.5 )
            )

P = smoothness_prior(model; p = 4)

solver = ACEfit.{solver}()

acefit!(model, data; solver=solver, weights=weights, prior = P, data_keys...)

@info("Training Error Table")
ACEpotentials.linear_errors(data, model; data_keys...)

@info("Testing Error Table")
ACEpotentials.linear_errors(test_data, model; data_keys...)

@info("Manual RMSE Test")
potential = model.potential
train_energies = [ JuLIP.get_data(at, "{energy_name}") / length(at) for at in data]
model_energies_train = [energy(potential, at) / length(at) for at in data]
rmse_energy_train = norm(train_energies - model_energies_train) / sqrt(length(data))
test_energies = [ JuLIP.get_data(at, "{energy_name}") / length(at) for at in test_data]
model_energies_pred = [energy(potential, at) / length(at) for at in test_data]
rmse_energy_test = norm(test_energies - model_energies_pred) / sqrt(length(test_data))

df = DataFrame(rmse_energy_train = rmse_energy_train, rmse_energy_test = rmse_energy_test)
CSV.write("rmse_energies.csv", df)

save_potential("acemodel.json", model)
export2lammps("acemodel.yace", model)
    '''

    # processed_text = '\n'.join(line.lstrip() for line in ace_text.split('\n'))

    with open('ace.jl', "w") as file:
        file.write(ace_text)

    run_command(f'export OMP_NUM_THREADS={num_of_threads} && julia --project=/home/epsilon/vld/iclb0745/.julia/environments/v1.9/ ace.jl')

    df = pd.read_csv("rmse_energies.csv")
    train_error = df['rmse_energy_train'][0]
    test_error = df['rmse_energy_test'][0]
    
    return train_error, test_error



########## fitting using Nequip ##########

def nequip_fitting(dir=None,
                  ):
                   

    nequip_text = f'''root: results
run_name: autoplex
seed: 123
dataset_seed: 456            
append: true 
default_dtype: float64

# network
r_max: 4.0
num_layers: 4
l_max: 2
parity: true
num_features: 32
nonlinearity_type: gate

nonlinearity_scalars:
  e: silu
  o: tanh

nonlinearity_gates:
  e: silu
  o: tanh

num_basis: 8
BesselBasis_trainable: true
PolynomialCutoff_p: 6

invariant_layers: 2
invariant_neurons: 64
avg_num_neighbors: auto

use_sc: true
dataset: ase
dataset_file_name: ./

key_mapping:
  z: atomic_numbers
  E: total_energy
  F: forces
  R: pos 

chemical_symbols:
  - H

wandb: true
wandb_project: autoplex

verbose: info
log_batch_freq: 10
log_epoch_freq: 1
save_checkpoint_freq: -1
save_ema_checkpoint_freq: -1

n_train: 100
n_val: 50
learning_rate: 0.005
batch_size: 5
validation_batch_size: 10
max_epochs: 10000
train_val_split: sequential
shuffle: true
metrics_key: validation_loss
use_ema: true
ema_decay: 0.99
ema_use_num_updates: true
report_init_validation: true

early_stopping_patiences:
  validation_loss: 50

early_stopping_lower_bounds: 
  LR: 1.0e-5

loss_coeffs:
  forces: 1
  total_energy:
    - 1
    - PerAtomMSELoss

metrics_components:
  - - forces
    - mae
  - - forces
    - rmse
  - - forces
    - mae
    - PerSpecies: True
      report_per_component: False          
  - - forces                                
    - rmse                                  
    - PerSpecies: True                     
      report_per_component: False    
  - - total_energy
    - mae    
  - - total_energy
    - mae
    - PerAtom: True

optimizer_name: Adam
optimizer_amsgrad: true

lr_scheduler_name: ReduceLROnPlateau
lr_scheduler_patience: 100
lr_scheduler_factor: 0.5

per_species_rescale_shifts_trainable: false
per_species_rescale_scales_trainable: false

per_species_rescale_shifts: dataset_per_atom_total_energy_mean
per_species_rescale_scales: dataset_forces_rms
    '''

    with open('nequip.yaml', "w") as file:
        file.write(nequip_text)


# train_error, test_error = ace_fitting(isol_es={14: -0.81432699}, 
#                                       order=2,
#                                       totaldegree=6,
#                                       cutoff=2.0,
#                                       num_of_threads=1)


# print(train_error)
# print(test_error)

# nequip_fitting(dir=None)
# with open('nequip.yaml', 'r') as file:
#     data = yaml.safe_load(file)


# data['r_max'] = 15.0  
# if 'chemical_symbols' in data:
#     if 'H' in data['chemical_symbols']:
#         data['chemical_symbols'] = [sym if sym != 'H' else 'As' for sym in data['chemical_symbols']]
#     data['chemical_symbols'].append('Si')


# data['key_mapping']['E'] = 'REF_energy'


# with open('nequip.yaml', 'w') as file:
#     yaml.safe_dump(data, file)