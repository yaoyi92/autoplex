"""Settings for autoplex."""

from __future__ import annotations

from typing import Any, Literal

from monty.serialization import loadfn
from pydantic import BaseModel, ConfigDict, Field


class UpdateBaseModel(BaseModel):
    """Base class for all models in autoplex."""

    model_config = ConfigDict(validate_assignment=True, protected_namespaces=())

    def update_parameters(self, updates: dict[str, Any]):
        """
        Update the default parameters of the model instance, including nested fields.

        Args:
            updates (Dict[str, Any]): A dictionary containing the fields as keys to update.
        """
        for key, value in updates.items():
            if hasattr(self, key):
                field_value = getattr(self, key)
                if isinstance(field_value, self.__class__) and isinstance(value, dict):
                    # Update nested model
                    field_value.update_parameters(
                        value
                    )  # Recursively call update_fields
                else:
                    # Update field value
                    setattr(self, key, value)

            # else:
            #    raise ValueError(
            #        f"Field {key} does not exist in {self.__class__.__name__}."
            #    )

    @classmethod
    def from_file(cls, filename: str):
        """
        Load the parameters from a file.

        Args:
            filename (str): The name of the file to load the parameters from.
        """
        custom_params = loadfn(filename)
        return cls(**custom_params)


class GeneralSettings(UpdateBaseModel):
    """Model describing general hyperparameters for the GAP fits."""

    at_file: str = Field(
        default="train.extxyz", description="Name of the training file"
    )
    default_sigma: str = Field(
        default="{0.0001 0.05 0.05 0}", description="Default sigma values"
    )
    energy_parameter_name: str = Field(
        default="REF_energy", description="Name of the energy parameter"
    )
    force_parameter_name: str = Field(
        default="REF_forces", description="Name of the force parameter"
    )
    virial_parameter_name: str = Field(
        default="REF_virial", description="Name of the virial parameter"
    )
    sparse_jitter: float = Field(default=1.0e-8, description="Sparse jitter")
    do_copy_at_file: str = Field(default="F", description="Copy the training file to")
    openmp_chunk_size: int = Field(default=10000, description="OpenMP chunk size")
    gp_file: str = Field(default="gap_file.xml", description="Name of the GAP file")
    e0_offset: float = Field(default=0.0, description="E0 offset")
    two_body: bool = Field(
        default=False, description="Whether to include two-body terms"
    )
    three_body: bool = Field(
        default=False, description="Whether to include three-body terms"
    )
    soap: bool = Field(default=True, description="Whether to include SOAP terms")


class TwobSettings(UpdateBaseModel):
    """Model describing two body hyperparameters for the GAP fits."""

    distance_Nb_order: int = Field(
        default=2,
        description="Distance_Nb order for two-body",
        alias="distance_Nb order",
    )
    f0: float = Field(default=0.0, description="F0 value for two-body")
    add_species: str = Field(
        default="T", description="Whether to add species information"
    )
    cutoff: float | int = Field(default=5.0, description="Radial cutoff distance")
    n_sparse: int = Field(default=15, description="Number of sparse points")
    covariance_type: str = Field(
        default="ard_se", description="Covariance type for two-body"
    )
    delta: float = Field(default=2.00, description="Delta value for two-body")
    theta_uniform: float = Field(
        default=0.5, description="Width of the uniform distribution for theta"
    )
    sparse_method: str = Field(
        default="uniform", description="Sparse method for two-body"
    )
    compact_clusters: str = Field(
        default="T", description="Whether to compact clusters"
    )


class ThreebSettings(UpdateBaseModel):
    """Model describing threebody hyperparameters for the GAP fits."""

    distance_Nb_order: int = Field(
        default=3,
        description="Distance_Nb order for three-body",
        alias="distance_Nb order",
    )
    f0: float = Field(default=0.0, description="F0 value for three-body")
    add_species: str = Field(
        default="T", description="Whether to add species information"
    )
    cutoff: float | int = Field(default=3.25, description="Radial cutoff distance")
    n_sparse: int = Field(default=100, description="Number of sparse points")
    covariance_type: str = Field(
        default="ard_se", description="Covariance type for three-body"
    )
    delta: float = Field(default=2.00, description="Delta value for three-body")
    theta_uniform: float = Field(
        default=1.0, description="Width of the uniform distribution for theta"
    )
    sparse_method: str = Field(
        default="uniform", description="Sparse method for three-body"
    )
    compact_clusters: str = Field(
        default="T", description="Whether to compact clusters"
    )


class SoapSettings(UpdateBaseModel):
    """Model describing soap hyperparameters for the GAP fits."""

    add_species: str = Field(
        default="T", description="Whether to add species information"
    )
    l_max: int = Field(default=10, description="Maximum degree of spherical harmonics")
    n_max: int = Field(
        default=12, description="Maximum number of radial basis functions"
    )
    atom_sigma: float = Field(default=0.5, description="Width of Gaussian smearing")
    zeta: int = Field(default=4, description="Exponent for dot-product SOAP kernel")
    cutoff: float = Field(default=5.0, description="Radial cutoff distance")
    cutoff_transition_width: float = Field(
        default=1.0, description="Width of the transition region for the cutoff"
    )
    central_weight: float = Field(default=1.0, description="Weight for central atom")
    n_sparse: int = Field(default=6000, description="Number of sparse points")
    delta: float = Field(default=1.00, description="Delta value for SOAP")
    f0: float = Field(default=0.0, description="F0 value for SOAP")
    covariance_type: str = Field(
        default="dot_product", description="Covariance type for SOAP"
    )
    sparse_method: str = Field(
        default="cur_points", description="Sparse method for SOAP"
    )


class GAPSettings(UpdateBaseModel):
    """Model describing the hyperparameters for the GAP fits for Phonons."""

    general: GeneralSettings = Field(
        default_factory=GeneralSettings,
        description="General hyperparameters for the GAP fits",
    )
    twob: TwobSettings = Field(
        default_factory=TwobSettings,
        description="Two body hyperparameters for the GAP fits",
    )
    threeb: ThreebSettings = Field(
        default_factory=ThreebSettings,
        description="Three body hyperparameters for the GAP fits",
    )
    soap: SoapSettings = Field(
        default_factory=SoapSettings,
        description="Soap hyperparameters for the GAP fits",
    )


class JACESettings(UpdateBaseModel):
    """Model describing the hyperparameters for the J-ACE fits."""

    order: int = Field(default=3, description="Order of the J-ACE model")
    totaldegree: int = Field(default=6, description="Total degree of the J-ACE model")
    cutoff: float = Field(default=2.0, description="Radial cutoff distance")
    solver: str = Field(default="BLR", description="Solver for the J-ACE model")


class NEQUIPSettings(UpdateBaseModel):
    """Model describing the hyperparameters for the NEQUIP fits."""

    r_max: float = Field(default=4.0, description="Radial cutoff distance")
    num_layers: int = Field(default=4, description="Number of layers")
    l_max: int = Field(default=2, description="Maximum degree of spherical harmonics")
    num_features: int = Field(default=32, description="Number of features")
    num_basis: int = Field(default=8, description="Number of basis functions")
    invariant_layers: int = Field(default=2, description="Number of invariant layers")
    invariant_neurons: int = Field(
        default=64, description="Number of invariant neurons"
    )
    batch_size: int = Field(default=5, description="Batch size")
    learning_rate: float = Field(default=0.005, description="Learning rate")
    max_epochs: int = Field(default=10000, description="Maximum number of epochs")
    default_dtype: str = Field(default="float32", description="Default data type")


class M3GNETSettings(UpdateBaseModel):
    """Model describing the hyperparameters for the M3GNET fits."""

    exp_name: str = Field(default="training", description="Name of the experiment")
    results_dir: str = Field(
        default="m3gnet_results", description="Directory to save the results"
    )
    cutoff: float = Field(default=5.0, description="Radial cutoff distance")
    threebody_cutoff: float = Field(
        default=4.0, description="Three-body cutoff distance"
    )
    batch_size: int = Field(default=10, description="Batch size")
    max_epochs: int = Field(default=1000, description="Maximum number of epochs")
    include_stresses: bool = Field(
        default=True, description="Whether to include stresses"
    )
    hidden_dim: int = Field(default=128, description="Hidden dimension")
    num_units: int = Field(default=128, description="Number of units")
    max_l: int = Field(default=4, description="Maximum degree of spherical harmonics")
    max_n: int = Field(
        default=4, description="Maximum number of radial basis functions"
    )
    test_equal_to_val: bool = Field(
        default=True, description="Whether the test set is equal to the validation set"
    )


class MACESettings(UpdateBaseModel):
    """Model describing the hyperparameters for the MACE fits."""

    model: str = Field(default="MACE", description="type of the model")
    name: str = Field(default="MACE_model", description="Name of the model")
    config_type_weights: str = Field(
        default="{'Default':1.0}", description="Weights for the configuration types"
    )
    hidden_irreps: str = Field(default="128x0e + 128x1o", description="Hidden irreps")
    r_max: float = Field(default=5.0, description="Radial cutoff distance")
    batch_size: int = Field(default=10, description="Batch size")
    max_num_epochs: int = Field(default=1500, description="Maximum number of epochs")
    start_swa: int = Field(default=1200, description="Start of the SWA")
    ema_decay: float = Field(
        default=0.99, description="Exponential moving average decay"
    )
    correlation: int = Field(default=3, description="Correlation")
    loss: str = Field(default="huber", description="Loss function")
    default_dtype: str = Field(default="float32", description="Default data type")
    swa: bool = Field(default=True, description="Whether to use SWA")
    ema: bool = Field(default=True, description="Whether to use EMA")
    amsgrad: bool = Field(default=True, description="Whether to use AMSGrad")
    restart_latest: bool = Field(
        default=True, description="Whether to restart the latest model"
    )
    seed: int = Field(default=123, description="Seed for the random number generator")
    device: Literal["cpu", "cuda"] = Field(
        default="cpu", description="Device to be used for model fitting"
    )


class NEPSettings(UpdateBaseModel):
    """Model describing the hyperparameters for the NEP fits."""

    version: int = Field(default=4, description="Version of the NEP model")
    type: list[int | str] = Field(
        default_factory=lambda: [1, "X"],
        description="Mandatory Parameter. Number of atom types and list of "
        "chemical species. Number of atom types must be an integer, followed by "
        "chemical symbols of species as in periodic table "
        "for which model needs to be trained, separated by comma. "
        "Default is [1, 'X'] as a placeholder. Example: [2, 'Pb', 'Te']",
    )
    type_weight: float = Field(
        default=1.0, description="Weights for different chemical species"
    )
    model_type: int = Field(
        default=0,
        description="Type of model that is being trained. "
        "Can be 0 (potential), 1 (dipole), "
        "2 (polarizability)",
    )
    prediction: int = Field(
        default=0, description="Mode of NEP run. Set 0 for training and 1 for inference"
    )
    cutoff: list[int, int] = Field(
        default_factory=lambda: [6, 5],
        description="Radial and angular cutoff. First element is for radial cutoff "
        "and second element is for angular cutoff",
    )
    n_max: list[int, int] = Field(
        default_factory=lambda: [4, 4],
        description="Number of radial and angular descriptors. First element "
        "is for radial and second element is for angular.",
    )
    basis_size: list[int, int] = Field(
        default_factory=lambda: [8, 8],
        description="Number of basis functions that are used to build the radial and angular descriptor. "
        "First element is for radial descriptor and second element is for angular descriptor",
    )
    l_max: list[int] = Field(
        default_factory=lambda: [4, 2, 1],
        description="The maximum expansion order for the angular terms. "
        "First element is for three-body, second element is for four-body and third element is for five-body",
    )
    neuron: int = Field(
        default=80, description="Number of neurons in the hidden layer."
    )
    lambda_1: float = Field(
        default=0.0, description="Weight for the L1 regularization term."
    )
    lambda_e: float = Field(default=1.0, description="Weight for the energy loss term.")
    lambda_f: float = Field(default=1.0, description="Weight for the force loss term.")
    lambda_v: float = Field(default=0.1, description="Weight for the virial loss term.")
    force_delta: int = Field(
        default=0,
        description=" Sets bias the on the loss function to put more emphasis "
        "on obtaining accurate predictions for smaller forces.",
    )
    batch: int = Field(default=1000, description="Batch size for training.")
    population: int = Field(
        default=60, description="Size of the population used by the SNES algorithm."
    )
    generation: int = Field(
        default=100000, description="Number of generations used by the SNES algorithm."
    )
    zbl: int = Field(
        default=2,
        description="Cutoff to use in universal ZBL potential at short distances. "
        "Acceptable values are in range 1 to 2.5.",
    )


class MLIPHypers(UpdateBaseModel):
    """Model containing the hyperparameter defaults for supported MLIPs in autoplex."""

    GAP: GAPSettings = Field(
        default_factory=GAPSettings, description="Hyperparameters for the GAP model"
    )
    J_ACE: JACESettings = Field(
        default_factory=JACESettings, description="Hyperparameters for the J-ACE model"
    )
    NEQUIP: NEQUIPSettings = Field(
        default_factory=NEQUIPSettings,
        description="Hyperparameters for the NEQUIP model",
    )
    M3GNET: M3GNETSettings = Field(
        default_factory=M3GNETSettings,
        description="Hyperparameters for the M3GNET model",
    )
    MACE: MACESettings = Field(
        default_factory=MACESettings, description="Hyperparameters for the MACE model"
    )
    NEP: NEPSettings = Field(
        default_factory=NEPSettings, description="Hyperparameters for the NEP model"
    )


# RSS Configuration


class ResumeFromPreviousState(UpdateBaseModel):
    """
    A model describing the state information.

    Useful to resume a previously interrupted or saved RSS workflow.
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


class SoapParas(UpdateBaseModel):
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


class BcurParams(UpdateBaseModel):
    """A model describing the parameters for the BCUR method."""

    soap_paras: SoapParas = Field(default_factory=SoapParas)
    frac_of_bcur: float = Field(
        default=0.8, description="Fraction of Boltzmann CUR selections"
    )
    bolt_max_num: int = Field(
        default=3000, description="Maximum number of Boltzmann selections"
    )


class BuildcellOptions(UpdateBaseModel):
    """A model describing the parameters for buildcell."""

    NFORM: str | None = Field(default=None, description="The number of formula units")
    SYMMOPS: str | None = Field(default=None, description="The symmetry operations")
    SLACK: float | None = Field(default=None, description="The slack factor")
    OVERLAP: float | None = Field(default=None, description="The overlap factor")
    MINSEP: str | None = Field(default=None, description="The minimum separation")


class CustomIncar(UpdateBaseModel):
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


class Twob(UpdateBaseModel):
    """A model describing the two-body GAP parameters."""

    cutoff: float = Field(default=5.0, description="Radial cutoff distance")
    n_sparse: int = Field(default=15, description="Number of sparse points")
    theta_uniform: float = Field(
        default=1.0, description="Width of the uniform distribution for theta"
    )


class Threeb(UpdateBaseModel):
    """A model describing the three-body GAP parameters."""

    cutoff: float = Field(default=3.0, description="Radial cutoff distance")


class Soap(UpdateBaseModel):
    """A model describing the SOAP GAP parameters."""

    l_max: int = Field(default=10, description="Maximum degree of spherical harmonics")
    n_max: int = Field(
        default=10, description="Maximum number of radial basis functions"
    )
    atom_sigma: float = Field(default=0.5, description="Width of Gaussian smearing")
    n_sparse: int = Field(default=2500, description="Number of sparse points")
    cutoff: float = Field(default=5.0, description="Radial cutoff distance")


class General(UpdateBaseModel):
    """A model describing the general GAP parameters."""

    three_body: bool = Field(
        default=False, description="Whether to include three-body terms"
    )


class RssConfig(UpdateBaseModel):
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
