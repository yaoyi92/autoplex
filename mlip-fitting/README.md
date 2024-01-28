For using this code for fitting MLIPs, one could use the following two jobs:

    job1 = data_preprocessing(split_ratio=0.1, regularization=True, distillation=True, f_max=40).make(vasp_ref_dir=path-to-vasp-ref, pre_database_dir=None)
    job2 = mlip_fit(mlip_type='GAP').make(database_dir=job1.output, isol_es=None)

The "split_ratio" parameter is used to divide the training set and the test set. A value of 0.1 means that the ratio of the training set to the test set is 9:1.

The name of the vasp reference dataset file needs to be "vasp_ref.extxyz". The energy of isolated atoms should already be included in "vasp_ref.extxyz", and the program will automatically read this information.
