from config import TrainConfig, Config, ArtifactConfig, SearchConfig
from dsl import DSL
from logger.logger import logger
from vae.models import load_model
from utils.models import load_model_from_template
from search.latent_search import LatentSearch
from search.encoder_only_search import LatentSearch as NeuralLatentSearch
from tasks import get_task_cls
from aim import Run
import jax
from utils.rng import make_key_gen


    task_cls = get_task_cls(Config.env_task)
    params = torch.load(Config.model_params_path(), map_location=Runtime.device)
    model.load_state_dict(params, strict=False)
    searcher = get_latent_search()(model, task_cls, dsl)

    StdoutLogger.log(
        "Main",
        f"Starting Latent Search with model {Config.model_name} for task {Config.env_task}",
    )

    best_program, converged, num_evaluations = searcher.search()

    StdoutLogger.log("Main", f"Converged: {converged}")
    StdoutLogger.log("Main", f"Final program: {best_program}")
    StdoutLogger.log("Main", f"Number of evaluations: {num_evaluations}")
