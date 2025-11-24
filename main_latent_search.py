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
if __name__ == '__main__':
    
    dsl = DSL.init_default_karel()
    
    
    keygen = make_key_gen(jax.random.key(SearchConfig.seed))

    model_template = load_model(TrainConfig.model_name, dsl) #This loads the PyTree
    model = load_model_from_template(ArtifactConfig.model_params_path, model_template)

    task_cls = get_task_cls(Config.env_task)
    run = Run()
    searcher = LatentSearch(model, task_cls, dsl, run)
    
    logger.info('Main', f'Starting Latent Search with model {TrainConfig.model_name} for task {Config.env_task}')
    
    search_state = searcher.search(key = next(keygen))
    
    logger.info("Search Done")
