from autoplex.data.common.jobs import collect_dft_data
from autoplex.data.common.flows import DFTStaticLabelling
from ase.io import read
from pymatgen.core.structure import Structure
from atomate2.settings import Atomate2Settings
from jobflow import run_locally, Flow


def test_vasp_static(test_dir, mock_vasp, memory_jobstore, clean_dir):
    settings = Atomate2Settings()

    poscar_paths = {
        f"static_bulk_{i}": test_dir / f"vasp/rss/Si_bulk_{i + 1}/inputs/POSCAR"
        for i in range(18)
    }

    test_structures = []
    for path in poscar_paths.values():
        structure = Structure.from_file(path)
        test_structures.append(structure)

    ref_paths = {
        **{f"static_bulk_{i}": f"rss/Si_bulk_{i + 1}/" for i in range(18)},
        "static_isolated_0": "rss/Si_isolated_1/",
        "static_dimer_0": "rss/Si_dimer_1/",
        "static_dimer_1": "rss/Si_dimer_2/",
        "static_dimer_2": "rss/Si_dimer_3/",
    }

    fake_run_vasp_kwargs = {
        "static_isolated_0": {"incar_settings": {"ISPIN": 2, "KSPACINGS": 2.0}},
        "static_dimer_0": {"incar_settings": {"ISPIN": 2, "KSPACINGS": 2.0}},
        "static_dimer_1": {"incar_settings": {"ISPIN": 2, "KSPACINGS": 2.0}},
        "static_dimer_2": {"incar_settings": {"ISPIN": 2, "KSPACINGS": 2.0}},
    }

    mock_vasp(ref_paths, fake_run_vasp_kwargs)

    job_dft = DFTStaticLabelling(isolated_atom=True,
                                 e0_spin=True,
                                 isolatedatom_box=[20.0, 20.5, 21.0],
                                 dimer=True,
                                 dimer_box=[15.0, 15.5, 16.0],
                                 dimer_range=[1.5, 2.0],
                                 dimer_num=3,
                                 custom_incar={
                                     "ADDGRID": None,
                                     "ENCUT": 200,
                                     "EDIFF": 1E-04,
                                     "ISMEAR": 0,
                                     "SIGMA": 0.05,
                                     "PREC": "Normal",
                                     "ISYM": None,
                                     "KSPACING": 0.3,
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
                                     "NELM": 50,
                                     "LMAXMIX": None,
                                     "LASPH": None,
                                     "AMIN": None,
                                 },
                                 ).make(structures=test_structures)

    job_collect_data = collect_dft_data(vasp_dirs=job_dft.output)

    response = run_locally(
        Flow([job_dft, job_collect_data]),
        create_folders=True,
        ensure_success=True,
        store=memory_jobstore
    )

    dict_vasp = job_collect_data.output.resolve(memory_jobstore)

    path_to_vasp, isol_energy = dict_vasp['vasp_ref_dir'], dict_vasp['isolated_atom_energies']

    atoms = read(path_to_vasp, index=":")
    config_types = [at.info['config_type'] for at in atoms]

    assert isol_energy['14'] == -0.84696938
    assert len(config_types) == 22
    assert 'IsolatedAtom' in config_types
    assert config_types.count("dimer") == 3
    assert config_types.count("bulk") == 18