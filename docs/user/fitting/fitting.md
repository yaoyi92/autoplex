(fitting)=

# Fitting potentials

This tutorial will show you how to control the MLIP fit settings with the `autoplex` workflow. The choice of the correct fit setup and hyperparameter settings has a significant influence on the final result.

There are two categories of fit settings that you can change. The first type concerns the general fit setup, that will affect the fit regardless of the chosen MLIP method, and e.g. changes database specific settings (like the split-up into training and test data). The other type of settings influences the MLIP specific setup like e.g. the choice of hyperparameters.

```python
complete_flow = CompleteDFTvsMLBenchmarkWorkflow(ml_models=["GAP", "MACE"], mlip_hyper=).make(...,
                                                                               f_max=40.0,
                                                                               fit_kwargs={
                                                                                   "split_ratio": 0.01,
                                                                                   "regularization": False,
                                                                                   "separated": True,
                                                                                   "num_processes": 48,
                                                                               })
```

## GAP

```python
complete_flow = CompleteDFTvsMLBenchmarkWorkflow(..., HPloop=False, 
                                                 atomwise_regularization_list=[0.01, 0.1], 
                                                 soap_delta_list=[0.5, 1.0, 1.5], 
                                                 n_sparse_list=[1000, 3000, 6000, 9000]).make(..., 
                                                                      atomwise_regularization_parameter=, 
                                                                      atom_wise_regularization= , 
                                                                      f_min=, 
                                                                      auto_delta=,
                                                                      **{...,
                                                                         "twob": {"cutoff": 5.0,...},
                                                                         "threeb": {"cutoff": 5.0,...},
                                                                         "general": {"default_sigma": "{0.001 0.05 0.05 0.0}",...},
                                                                         "soap": {"delta": 1.0, "l_max": 12, "n_max": 10,...},
        
    })
```
    "at_file": "trainGAP.xyz",
    "energy_parameter_name": "REF_energy",
    "force_parameter_name": "REF_forces",
    "virial_parameter_name": "REF_virial",
    "gp_file": "gap_file.xml"

gap-defaults.json

## ACE

## Nequip

## M3GNet

## MACE
