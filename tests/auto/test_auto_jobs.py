from __future__ import annotations

from pymatgen.core.structure import Structure
from autoplex.auto.jobs import get_phonon_ml_calculation_jobs
from atomate2.common.schemas.phonons import PhononBSDOSDoc


def test_get_phonon_ml_calculation_jobs(test_dir, clean_dir, memory_jobstore):
    from jobflow import run_locally

    potential_file_dir = test_dir / "gap" / "gap.xml"
    path_to_struct = test_dir / "gap" / "POSCAR"
    structure = Structure.from_file(path_to_struct)

    gap_phonon_jobs = get_phonon_ml_calculation_jobs(
        ml_dir=potential_file_dir, structure=structure, min_length=20
    )

    responses = run_locally(
        gap_phonon_jobs, create_folders=True, ensure_success=True, store=memory_jobstore
    )

    ml_phonon_bs_doc = responses[gap_phonon_jobs.output.uuid][2].output.resolve(
        store=memory_jobstore
    )
    assert isinstance(ml_phonon_bs_doc, PhononBSDOSDoc)
