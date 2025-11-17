import os 
import sys


# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

### fix this to make it work with tyro:

import torch
from config import Config
from src.config.runtime import Runtime
from logger.stdout_logger import StdoutLogger
from search.latent_search import LatentSearch
from search.encoder_only_search import LatentSearch as NeuralLatentSearch
from tasks import get_task_cls

###TODO: find a way to make this more modular:
def get_latent_search():
    if "Leaps" in Config.model_name:
        return LatentSearch
    else:
        return NeuralLatentSearch

if __name__ == "__main__":

    model, device, dsl = Runtime.get_policy_model(), Runtime.device, Runtime.dsl

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
