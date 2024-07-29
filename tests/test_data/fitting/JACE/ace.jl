using ACEpotentials
using LinearAlgebra: norm, Diagonal
using CSV, DataFrames
using Distributed
addprocs(3, exeflags="--project=$(Base.active_project())")
@everywhere using ACEpotentials

data_file = "train_ace.extxyz"
data = read_extxyz(data_file)
test_data_file = "test.extxyz"
test_data = read_extxyz(test_data_file)
data_keys = (energy_key = "REF_energy", force_key = "REF_force", virial_key = "REF_virial")

model = acemodel(elements=[:Si],
                order=3,
                totaldegree=6,
                rcut=2.0,
                Eref=[:Si => -0.84696938])

weights = Dict(
            "crystal" => Dict("E" => 30.0, "F" => 1.0 , "V" => 1.0 ),
            "RSS" => Dict("E" => 3.0, "F" => 0.5 , "V" => 0.1 ),
            "amorphous" => Dict("E" => 3.0, "F" => 0.5 , "V" => 0.1 ),
            "liquid" => Dict("E" => 10.0, "F" => 0.5 , "V" => 0.25 ),
            "RSS_initial" => Dict("E" => 1.0, "F" => 0.5 , "V" => 0.1 ),
            "dimer" => Dict("E" => 30.0, "F" => 1.0 , "V" => 1.0 )
            )

P = smoothness_prior(model; p = 4)

solver = ACEfit.BLR()

acefit!(model, data; solver=solver, weights=weights, prior = P, data_keys...)

@info("Training Error Table")
ACEpotentials.linear_errors(data, model; data_keys...)

@info("Testing Error Table")
ACEpotentials.linear_errors(test_data, model; data_keys...)

@info("Manual RMSE Test")
potential = model.potential
train_energies = [ JuLIP.get_data(at, "REF_energy") / length(at) for at in data]
model_energies_train = [energy(potential, at) / length(at) for at in data]
rmse_energy_train = norm(train_energies - model_energies_train) / sqrt(length(data))
test_energies = [ JuLIP.get_data(at, "REF_energy") / length(at) for at in test_data]
model_energies_pred = [energy(potential, at) / length(at) for at in test_data]
rmse_energy_test = norm(test_energies - model_energies_pred) / sqrt(length(test_data))

df = DataFrame(rmse_energy_train = rmse_energy_train, rmse_energy_test = rmse_energy_test)
CSV.write("rmse_energies.csv", df)

save_potential("acemodel.json", model)
export2lammps("acemodel.yace", model)
    