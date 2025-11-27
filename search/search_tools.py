import jax.numpy as jnp
from logger.logger import logger
from dsl import DSL
from config import SearchConfig, TrainConfig
from dataclasses import dataclass
from functools import partial
from typing import List, Callable
from utils.program import execute_program 
from multiprocessing import Pool
from vae.models.base_vae import BaseVAE
import jax


@dataclass
class SearchState:

    population: jax.Array
    converged: bool
    counter_for_restart: int
    best_reward: float
    best_program: str
    prev_mean_elite_reward: float
    sigma: float
    num_evaluations: int = 0

def get_search_method(method: str) -> Callable:
    if method == "CEM":
        if SearchConfig.cem_reduce_to_mean:
            return CEM_MEAN
        else: 
            return CEM
    elif method == "CEBS":
        return CEBS
    else: 
        raise ValueError(f"Expected `CEM` or `CEBS` as search methods, but got {method}")

def init_search_state(key: jax.Array) -> SearchState: 

    search_state = SearchState(population = init_population(TrainConfig.hidden_size, SearchConfig.population_size, key = key),
                                converged = False,
                                num_evaluations = 0,
                                counter_for_restart = 0,
                                best_reward = -float("inf"),
                                best_program = None,
                                prev_mean_elite_reward = -float("inf"),
                                sigma = SearchConfig.initial_sigma,
                                )

    return search_state 

def get_neighbors(sigma: float, elite_choice_indices:jax.Array, elite_population: jax.Array, key: jax.Array):
    mu = elite_population[elite_choice_indices]
    population = jax.random.normal(key, shape= mu.shape) + sigma
    return population

def search_sigma_anneal(sigma: float, sigma_min: float, sigma_rate: float):
    sigma = max(sigma_min, sigma * sigma_rate)
    return sigma

def init_population(latent_size: int, population_size: int, key: jax.Array) -> jax.Array:
    return jax.random.normal(key, shape = (population_size, latent_size))

### The logic is basically the same, in our version we are simply going to make sure 

def execute_population(
        model: BaseVAE, population: jax.Array, dsl: DSL, task_envs: List
    ) -> tuple[list[str], jax.Array, int]:
        """Runs the given population in the environment and returns a list of mean rewards, after
        `Config.search_number_executions` executions.

        Args:
            population (torch.Tensor): Current population as a tensor.

        Returns:
            tuple[list[str], int, torch.Tensor]: List of programs as strings, list of mean rewards
            as tensor and number of evaluations as int.
        """
        programs_tokens = model.decode_vector(population)
        ## At this step, the programs are turned into string (from tokens), and in order to be executed, they need to be turned into nodes. 
        programs_str = [
            dsl.parse_int_to_str(prog_tokens) for prog_tokens in programs_tokens
        ]
        
        
        #TODO: Learn to use/modify this to better use the multi-processing in the code.
        ### Use vmap here?

        ## BUG: This doesn't work in multi-processing
        if SearchConfig.multiprocessing and False:
            with Pool() as pool:
                fn = partial(execute_program, task_envs=task_envs, dsl=dsl)
                results = pool.map(fn, programs_str)
        else:
            results = [
                execute_program(p, task_envs, dsl) for p in programs_str
            ]
        return results


def process_population_results(results, dsl: DSL, search_state: SearchState):
    rewards = []
    # converged = False
    for p, num_eval, r in results:
        program_str = dsl.parse_node_to_str(p)
        rewards.append(r)
        search_state.num_evaluations += num_eval

        #Since the best reward is kept from previous runs, the best program is not changed between populations, unless a program that has even better return is found.
        if r > search_state.best_reward:
            search_state.best_reward = r
            best_program = program_str
            logger.info(
                    "Latent Search", f"New best reward: {search_state.best_reward}"
            )
            logger.info(
                "Latent Search", f"New best program: {best_program}"
            )
            logger.info(
                "Latent Search", f"Number of evaluations: {search_state.num_evaluations}"
            )

            ## For now I am not collecting the data. We will get to that with aim later.
            #collector.collect({'time':t, 'num_evaluations':num_evaluations, 'best_reward':best_reward})
            
        if search_state.best_reward >= 1.0:
            search_state.converged = True
            break

    return jnp.array(rewards), search_state


def maybe_continue_population(mean_elite_reward: float, 
                              search_state: SearchState, 
                              elite_population: jax.Array, 
                              search_method: Callable,
                              key: jax.Array):

    #Update counter
    search_state.counter_for_restart += mean_elite_reward == search_state.prev_mean_elite_reward
    if search_state.counter_for_restart >= SearchConfig.restart_timeout and SearchConfig.restart_timeout > 0:
                search_state.population = init_population()
                search_state.counter_for_restart = 0
                search_state.sigma = SearchConfig.initial_sigma
                # StdoutLogger.log("Latent Search", "Restarted population.")
    else:             
        elite_choice_indices, elite_population = search_method(elite_population, next(keygen))    
        search_state.population = get_neighbors(search_state.sigma, 
                                                elite_choice_indices, 
                                                elite_population,
                                                next(keygen))

        # Anneal Sigma only when actually continuing search?
        # Should I not actually reset sigma?
        # I should actually resent the sgima
        search_state.sigma = search_sigma_anneal(search_state.sigma, SearchConfig.sigma_min, SearchConfig.sigma_decay_rate)
             
    return search_state

def CEM_MEAN(elite_population: jax.Array, key: jax.Array):

    elite_choice_indices = jnp.zeros(SearchConfig.population_size, dtype = jnp.int16) #This just works!
    mean_elite = jnp.mean(elite_population, axis=0)       
    #Replace elite population with its mean
    elite_population = elite_population = jnp.tile(mean_elite, (SearchConfig.n_elite, 1))
    return elite_choice_indices, elite_population

def CEM(elite_population: jax.Array, key: jax.Array):
    ###choose n_elites randomly.
    elite_choice_indices = jax.random.choice(key, SearchConfig.n_elite, replace=True)
    return elite_choice_indices, elite_population

def CEBS(elite_population: jax.Array, key: jax.Array):
    elite_choice_indices = jnp.arange(SearchConfig.n_elite).repeat(int(SearchConfig.population_size, SearchConfig.n_elite))
    return elite_choice_indices, elite_population