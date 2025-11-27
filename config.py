from typing import Annotated, Callable
###NOTE: DO NOT change these defaults, somethings will break. 


PATH_TO_DATASETS = 'artifacts/datasets'

class TrainConfig:

    model_name: str = "LeapsVAE"
    seed: int = 0
    hidden_size: int = 256
    batch_size: int = 256
    num_epochs: int = 150
    prog_teacher_enforcing: bool = True
    a_h_teacher_enforcing: bool = True
    prog_loss_coeff: float = 1.0
    a_h_loss_coeff: float = 1.0
    latent_loss_coeff: float = 0.1
    learning_rate: float = 5e-4
    episode_length: int = 50


class SearchConfig:

    model_name = "LeapsVAESearch"
    seed: int = 0
    multiprocessing: bool = True
    initial_sigma: float = 1.0
    sigma_decay_rate: float = 0.9
    sigma_min: float =  0.1
    elitism_rate: float = 0.1
    population_size: int = 256
    num_env_executions: int = 16
    num_iterations: int = 1000
    restart_timeout: int = 5
    search_method_name: str = 'CEM'
    cem_reduce_to_mean: bool = False
    n_elite = int(elitism_rate * population_size)



class DataGenConfig: 
    
    num_programs: int = 50_000
    ratio_train: float = 0.7
    ratio_val: float = 0.15 
    ratio_test: float = 0.15
    

class ArtifactConfig:

    dataset_path: str =  "artifacts/data/programs.pkl"
    model_params_path: str = "artifacts/params/2025-11-25_00-36-47.pkl"
    

class Config:
    """Class that handles the project global configuration."""

    extension: Annotated[str, 'The file extension to save files to'] = 'csv'
   
    dsl_include_hole: Annotated[bool, "If set, DSL includes <HOLE> token."] = False

    experiment_name: Annotated[str, "Name of the model, used for saving output."] = (
        "program_vae"
    )

    multiprocessing_active: Annotated[
        bool, "If set, search functions will use multiprocessing to evaluate programs."
    ] = True

    model_name: Annotated[str, "Class name of the VAE model."] = "LeapsVAE"
    model_seed: Annotated[int, "Seed for model initialization."] = 1
    model_hidden_size: Annotated[int, "Number of dimensions in VAE hidden unit."] = 256

    search_sigma_exponential_decay: Annotated[float, "The exponential decay parameter"] = 0.9
    search_sigma_min: Annotated[float, "The minimum amount search sigma"] = 0.1
    assert search_sigma_exponential_decay > 0 and search_sigma_exponential_decay <= 1

    @classmethod
    def model_params_path(cls):
        "Path to model parameters. This is a property since it needs to be dynamically changed as the other properties change in run time."
        return f"params/{cls.model_name.split('VAE')[0].lower()}_vae_{cls.model_hidden_size}_{cls.trainer_latent_loss_coef}.ptp"

    datagen_num_programs: Annotated[
        int, "Number of programs in dataset, used for data generation and loading."
    ] = 50000
    datagen_sketch_iterations: Annotated[
        int,
        "Number of needed Top-Down iterations to reconstruct a program from its sketch",
    ] = 3
    datagen_generate_demos: Annotated[
        bool, "If set, generates demonstrations for each program."
    ] = False
    datagen_generate_sketches: Annotated[
        bool, "If set, generates sketches for each program."
    ] = False

    data_class_name: Annotated[str, "Name of program dataset class."] = "ProgramDataset"
    data_program_dataset_path: Annotated[str, "Path to program dataset."] = (
        "data/programs.pkl"
    )
    data_reduce_dataset: Annotated[
        bool, "Reduce dataset to 1000 samples for debugging"
    ] = False
    data_batch_size: Annotated[int, "Batch size used in VAE training."] = 256
    data_max_program_length: Annotated[
        int, "Maximum program length in number of tokens."
    ] = 45
    data_max_program_size: Annotated[
        int, "Maximum program size in number of nodes."
    ] = 20
    data_max_program_depth: Annotated[
        int, "Max allowed depth during program generation."
    ] = 4
    data_max_program_sequence: Annotated[
        int, "Max allowed number of sequential nodes aggregated by Conjunction."
    ] = 6
    data_max_demo_length: Annotated[
        int, "Maximum action history length in number of actions."
    ] = 100
    data_num_demo_per_program: Annotated[
        int, "Number of demonstrations per program in dataset."
    ] = 10
    data_ratio_train: Annotated[float, "Ratio of training data."] = 0.7
    data_ratio_val: Annotated[float, "Ratio of validation data."] = 0.15
    data_ratio_test: Annotated[float, "Ratio of test data."] = 0.15

    env_task: Annotated[str, "Name of Karel task to solve."] = "StairClimber"
    env_seed: Annotated[int, "Seed for random environment generation."] = 1


    @classmethod
    def env_height(cls):
        if cls.env_task =="CleanHouse":
            return  14
        else: 
            return 8
    @classmethod
    def env_width(cls):
        if cls.env_task =="CleanHouse":
            return  22
        else: 
            return 8
    # env_height: Annotated[int, "Height of Karel environment."] = 8
    # env_width: Annotated[int, "Width of Karel environment."] = 8
    env_enable_leaps_behaviour: Annotated[
        bool, "If set, uses LEAPS version of Karel rules."
    ] = False
    env_is_crashable: Annotated[bool, "If set, program stops when Karel crashes."] = (
        False
    )

    search_topdown_iterations: Annotated[
        int, "Maximum iterations for Top-Down Search."
    ] = 3
    search_elitism_rate: Annotated[
        float, "Elitism rate for selection phase of Latent Search."
    ] = 0.1
    search_population_size: Annotated[
        int, "Population size for growth phase of Latent Search."
    ] = 256
    search_method: Annotated[str, "The search method used to do the search"] = "CEM"
    search_sigma: Annotated[
        float, "Size of noise in growth phase of Latent Search."
    ] = 0.2
    search_number_executions: Annotated[
        int, "Number of environment executions for mean reward calculation."
    ] = 16
    search_number_iterations: Annotated[
        int, "Maximum number of iterations of Latent Search."
    ] = 1000
    search_restart_timeout: Annotated[
        int, "Maximum number of iterations without improvement before restart."
    ] = 5 # this is almost equalt to inf :)

    trainer_num_epochs: Annotated[int, "Number of training epochs."] = 150
    trainer_disable_prog_teacher_enforcing: Annotated[
        bool, "If set, program sequence classification will not use teacher enforcing."
    ] = False ### BUG: There is a bug here if set to true. 
    trainer_disable_a_h_teacher_enforcing: Annotated[
        bool, "If set, actions sequence classification will not use teacher enforcing."
    ] = False
    trainer_prog_loss_coef: Annotated[
        float, "Weight of program classification loss. Set to zero when trainig Project VAE"
    ] = 0.0
    trainer_a_h_loss_coef: Annotated[
        float, "Weight of actions classification loss."
    ] = 1.0
    trainer_latent_loss_coef: Annotated[float, "Weight of VAE KL Divergence Loss."] = (
        0.1
    )
    trainer_optim_lr: Annotated[float, "Adam optimizer learning rate."] = 5e-4
    trainer_save_params_each_epoch: Annotated[
        bool, "If set, trainer saves model params after each epoch."
    ] = False
    episode_length = 50
