"""Settings for autoplex."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ResumeFromPreviousState(BaseModel):
    """
    A model describing the state information.

    It is required to resume a previously interrupted or saved RSS workflow.
    When 'train_from_scratch' is set to False, this parameter is mandatory
    for the workflow to pick up from a saved state.
    """

    test_error: float | None = Field(
        default=None,
        description="The test error from the last completed training step.",
    )
    pre_database_dir: str | None = Field(
        default=None,
        description="Path to the directory containing the pre-existing database for resuming",
    )
    mlip_path: str | None = Field(
        default=None, description="Path to the file of a previous MLIP model."
    )
    isolated_atom_energies: dict | None = Field(
        default=None,
        description="A dictionary with isolated atom energy values mapped to atomic numbers",
    )


class SoapParas(BaseModel):
    """A model describing the SOAP parameters."""

    l_max: int = Field(default=12, description="Maximum degree of spherical harmonics")
    n_max: int = Field(
        default=12, description="Maximum number of radial basis functions"
    )
    atom_sigma: float = Field(default=0.0875, description="idth of Gaussian smearing")
    cutoff: float = Field(default=10.5, description="Radial cutoff distance")
    cutoff_transition_width: float = Field(
        default=1.0, description="Width of the transition region for the cutoff"
    )
    zeta: float = Field(default=4.0, description="Exponent for dot-product SOAP kernel")
    average: bool = Field(
        default=True, description="Whether to average the SOAP vectors"
    )
    species: bool = Field(
        default=True, description="Whether to consider species information"
    )


class BcurParams(BaseModel):
    """A model describing the parameters for the BCUR method."""

    soap_paras: SoapParas = Field(default_factory=SoapParas)
    frac_of_bcur: float = Field(
        default=0.8, description="Fraction of Boltzmann CUR selections"
    )
    bolt_max_num: int = Field(
        default=3000, description="Maximum number of Boltzmann selections"
    )


class BuildcellOptions(BaseModel):
    """A model describing the parameters for buildcell."""

    NFORM: str | None = Field(default=None, description="The number of formula units")
    SYMMOPS: str | None = Field(default=None, description="The symmetry operations")
    SLACK: float | None = Field(default=None, description="The slack factor")
    OVERLAP: float | None = Field(default=None, description="The overlap factor")
    MINSEP: str | None = Field(default=None, description="The minimum separation")


class CustomIncar(BaseModel):
    """A model describing the INCAR parameters."""

    ISMEAR: int = 0
    SIGMA: float = 0.05
    PREC: str = "Accurate"
    ADDGRID: str = ".TRUE."
    EDIFF: float = 1e-07
    NELM: int = 250
    LWAVE: str = ".FALSE."
    LCHARG: str = ".FALSE."
    ALGO: str = "Normal"
    AMIX: float | None = None
    LREAL: str = ".FALSE."
    ISYM: int = 0
    ENCUT: float = 520.0
    KSPACING: float = 0.2
    GGA: str | None = None
    KPAR: int = 8
    NCORE: int = 16
    LSCALAPACK: str = ".FALSE."
    LPLANE: str = ".FALSE."


class Twob(BaseModel):
    """A model describing the two-body GAP parameters."""

    cutoff: float = Field(default=5.0, description="Radial cutoff distance")
    n_sparse: int = Field(default=15, description="Number of sparse points")
    theta_uniform: float = Field(
        default=1.0, description="Width of the uniform distribution for theta"
    )


class Threeb(BaseModel):
    """A model describing the three-body GAP parameters."""

    cutoff: float = Field(default=3.0, description="Radial cutoff distance")


class Soap(BaseModel):
    """A model describing the SOAP GAP parameters."""

    l_max: int = Field(default=10, description="Maximum degree of spherical harmonics")
    n_max: int = Field(
        default=10, description="Maximum number of radial basis functions"
    )
    atom_sigma: float = Field(default=0.5, description="Width of Gaussian smearing")
    n_sparse: int = Field(default=2500, description="Number of sparse points")
    cutoff: float = Field(default=5.0, description="Radial cutoff distance")


class General(BaseModel):
    """A model describing the general GAP parameters."""

    three_body: bool = Field(
        default=False, description="Whether to include three-body terms"
    )


class RssConfig(BaseModel):
    """A model describing the complete RSS configuration."""

    tag: str | None = Field(
        default=None,
        description="Tag of systems. It can also be used for setting up elements "
        "and stoichiometry. For example, the tag of 'SiO2' will be recognized "
        "as a 1:2 ratio of Si to O and passed into the parameters of buildcell. "
        "However, note that this will be overwritten if the stoichiometric ratio "
        "of elements is defined in the 'buildcell_options'",
    )
    train_from_scratch: bool = Field(
        default=True,
        description="If True, it starts the workflow from scratch "
        "If False, it resumes from a previous state.",
    )
    resume_from_previous_state: ResumeFromPreviousState = Field(
        default_factory=ResumeFromPreviousState
    )
    generated_struct_numbers: list[int, int] = Field(
        default_factory=lambda: [8000, 2000],
        description="Expected number of generated "
        "randomized unit cells by buildcell.",
    )
    buildcell_options: list[BuildcellOptions] | None = Field(
        default=None, description="Customized parameters for buildcell."
    )
    fragment_file: str | None = Field(default=None, description="")
    fragment_numbers: list[int] | None = Field(
        default=None,
        description=" Numbers of each fragment to be included in the random structures. "
        "Defaults to 1 for all specified.",
    )
    num_processes_buildcell: int = Field(
        default=128, description="Number of processes for buildcell."
    )
    num_of_initial_selected_structs: list[int, int] = Field(
        default_factory=lambda: [80, 20],
        description="Number of structures to be sampled directly "
        "from the buildcell-generated randomized cells.",
    )
    num_of_rss_selected_structs: int = Field(
        default=100,
        description="Number of structures to be selected from each RSS iteration.",
    )
    initial_selection_enabled: bool = Field(
        default=True,
        description="If true, sample structures from initially generated "
        "randomized cells using CUR.",
    )
    rss_selection_method: Literal["bcur1s", "bcur2i", None] = Field(
        default="bcur2i",
        description="Method for selecting samples from the RSS trajectories: "
        "Boltzmann flat histogram in enthalpy first, then CUR. Options are as follows",
    )
    bcur_params: BcurParams = Field(
        default_factory=BcurParams, description="Parameters for the BCUR method."
    )
    random_seed: int | None = Field(
        default=None, description="A seed to ensure reproducibility of CUR selection."
    )
    include_isolated_atom: bool = Field(
        default=True,
        description="Perform single-point calculations for isolated atoms.",
    )
    isolatedatom_box: list[float, float, float] = Field(
        default_factory=lambda: [20.0, 20.0, 20.0],
        description="List of the lattice constants for an "
        "isolated atom configuration.",
    )
    e0_spin: bool = Field(
        default=False,
        description="Include spin polarization in isolated atom and dimer calculations",
    )
    include_dimer: bool = Field(
        default=True,
        description="Perform single-point calculations for dimers only once",
    )
    dimer_box: list[float, float, float] = Field(
        default_factory=lambda: [20.0, 20.0, 20.0],
        description="The lattice constants of a dimer box.",
    )
    dimer_range: list[float, float] = Field(
        default_factory=lambda: [1.0, 5.0],
        description="The range of the dimer distance.",
    )
    dimer_num: int = Field(
        default=21,
        description="Number of different distances to consider for dimer calculations.",
    )
    custom_incar: CustomIncar = Field(
        default_factory=CustomIncar,
        description="Custom VASP input parameters. "
        "If provided, will update the default parameters",
    )
    custom_potcar: str | None = Field(
        default=None,
        description="POTCAR settings to update. Keys are element symbols, "
        "values are the desired POTCAR labels.",
    )
    vasp_ref_file: str = Field(
        default="vasp_ref.extxyz", description="Reference file for VASP data"
    )
    config_types: list[str] = Field(
        default_factory=lambda: ["initial", "traj_early", "traj"],
        description="Configuration types for the VASP calculations",
    )
    rss_group: list[str] = Field(
        default_factory=lambda: ["traj"],
        description="Group of configurations for the RSS calculations",
    )
    test_ratio: float = Field(
        default=0.1,
        description="The proportion of the test set after splitting the data",
    )
    regularization: bool = Field(
        default=True,
        description="Whether to apply regularization. This only works for GAP to date.",
    )
    scheme: Literal["linear-hull", "volume-stoichiometry", None] = Field(
        default="linear-hull", description="Method to use for regularization"
    )
    reg_minmax: list[list[float]] = Field(
        default_factory=lambda: [
            [0.1, 1],
            [0.001, 0.1],
            [0.0316, 0.316],
            [0.0632, 0.632],
        ],
        description="List of tuples of (min, max) values for energy, force, "
        "virial sigmas for regularization",
    )
    distillation: bool = Field(
        default=False, description="Whether to apply data distillation"
    )
    force_max: float | None = Field(
        default=None, description="Maximum force value to exclude structures"
    )
    force_label: str | None = Field(
        default=None, description="The label of force values to use for distillation"
    )
    pre_database_dir: str | None = Field(
        default=None, description="Directory where the previous database was saved."
    )
    mlip_type: Literal["GAP", "J-ACE", "NEQUIP", "M3GNET", "MACE"] = Field(
        default="GAP", description="MLIP to be fitted"
    )
    ref_energy_name: str = Field(
        default="REF_energy", description="Reference energy name."
    )
    ref_force_name: str = Field(
        default="REF_forces", description="Reference force name."
    )
    ref_virial_name: str = Field(
        default="REF_virial", description="Reference virial name."
    )
    auto_delta: bool = Field(
        default=True,
        description="Whether to automatically calculate the delta value for GAP terms.",
    )
    num_processes_fit: int = Field(
        default=32, description="Number of processes used for fitting"
    )
    device_for_fitting: Literal["cpu", "cuda"] = Field(
        default="cpu", description="Device to be used for model fitting"
    )
    twob: Twob = Field(
        default_factory=Twob,
        description="Parameters for the two-body descriptor, Applicable on to GAP",
    )
    threeb: Threeb = Field(
        default_factory=Threeb,
        description="Parameters for the three-body descriptor, Applicable on to GAP",
    )
    soap: Soap = Field(
        default_factory=Soap,
        description="Parameters for the SOAP descriptor, Applicable on to GAP",
    )
    general: General = Field(
        default_factory=General, description="General parameters for the GAP model"
    )
    scalar_pressure_method: Literal["exp", "uniform"] = Field(
        default="uniform", description="Method for adding external pressures."
    )
    scalar_exp_pressure: int = Field(
        default=1, description="Scalar exponential pressure"
    )
    scalar_pressure_exponential_width: float = Field(
        default=0.2, description="Width for scalar pressure exponential"
    )
    scalar_pressure_low: int = Field(
        default=0, description="Lower limit for scalar pressure"
    )
    scalar_pressure_high: int = Field(
        default=25, description="Upper limit for scalar pressure"
    )
    max_steps: int = Field(
        default=300, description="Maximum number of steps for the GAP optimization"
    )
    force_tol: float = Field(
        default=0.01, description="Force residual tolerance for relaxation"
    )
    stress_tol: float = Field(
        default=0.01, description="Stress residual tolerance for relaxation."
    )
    stop_criterion: float = Field(
        default=0.01, description="Convergence criterion for stopping RSS iterations."
    )
    max_iteration_number: int = Field(
        default=25, description="Maximum number of RSS iterations to perform."
    )
    num_groups: int = Field(
        default=6,
        description="Number of structure groups, used for assigning tasks across multiple nodes."
        "For example, if there are 10,000 trajectories to relax and 'num_groups=10',"
        "the trajectories will be divided into 10 groups and 10 independent jobs will be created,"
        "with each job handling 1,000 trajectories.",
    )
    initial_kt: float = Field(
        default=0.3, description="Initial temperature (in eV) for Boltzmann sampling."
    )
    current_iter_index: int = Field(
        default=1, description="Current iteration index for the RSS."
    )
    hookean_repul: bool = Field(
        default=False, description="Whether to apply Hookean repulsion"
    )
    hookean_paras: dict | None = Field(
        default=None,
        description="Parameters for the Hookean repulsion as a "
        "dictionary of tuples.",
    )
    keep_symmetry: bool = Field(
        default=False, description="Whether to preserve symmetry during relaxations."
    )
    write_traj: bool = Field(
        default=True,
        description="Bool indicating whether to write the trajectory files.",
    )
    num_processes_rss: int = Field(
        default=128, description="Number of processes used for running RSS."
    )
    device_for_rss: Literal["cpu", "cuda"] = Field(
        default="cpu", description="Device to be used for RSS calculations."
    )
